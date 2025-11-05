"""
yolov8_model.py

    이 파일은 YOLOv8 세그멘테이션 모델을 로드하고,
    FastAPI의 main.py에서 이미지를 전달받아 모델 추론을 수행한 뒤,
    객체별 '감지 개수'와 '화면 내 실제 면적 비율(%)'을 계산해 반환합니다.

- 모델 로드는 서버 시작 시 단 1회만 수행됩니다.
- 추론은 run_inference(image_path) 함수로 실행됩니다.
"""


# ================ 검출 비율 수정!!!!!!!!!!!!!!!!!!! 버전 ================
from ultralytics import YOLO
import cv2, base64, os, numpy as np
import math

# 🎨 클래스별 색상 정의 (BGR)
CLASS_COLORS = {
    0: (0, 150, 15),   # 나무 (초록)
    1: (245, 35, 0),  # 비닐 (파랑)
    2: (38, 38, 245)  # 플라스틱 (빨강)
}

ALPHA = 0.3  # 마스크 투명도



# 🔷 고정된 실제 바닥 전체 면적 (10m 높이, FOV 60° x 45° 기준)
# 계산식: 바닥면적 = (2 * H * tan(HFOV/2)) * (2 * H * tan(VFOV/2))
# => 약 95.6m²
TOTAL_REAL_WORLD_AREA = 95.6  # 고정 상수 (m²)


class YOLOWrapper:
    def __init__(self, weight_path):
        self.model = YOLO(weight_path)
        self.names = self.model.names
        print(f"\n[YOLO] 모델 로드 완료")

        # 🔷 촬영 조건에 기반한 실제 바닥 전체 면적 (고정값)
        self.total_real_world_area = self._calculate_ground_area(fov_h=60, fov_v=45, height=10)

    def _calculate_ground_area(self, fov_h, fov_v, height):
        """
        카메라 화각과 촬영 높이를 기반으로 실제 바닥의 전체 촬영 면적(m²)을 계산
        """
        width_m = 2 * height * math.tan(math.radians(fov_h / 2))
        height_m = 2 * height * math.tan(math.radians(fov_v / 2))
        area = width_m * height_m
        return round(area, 2)  # 95.6m² 처럼 정리된 값 반환

    def _calculate_object_area(self, object_pixels, total_pixels):
        """
        YOLO segmentation mask에서 얻은 픽셀 수를 실제 면적(m²)과 비율(%)로 변환
        """
        if total_pixels == 0 or object_pixels == 0:
            return 0.0, 0.0

        pixel_ratio = object_pixels / total_pixels
        real_area = self.total_real_world_area * pixel_ratio
        percent = pixel_ratio * 100
        return round(real_area, 2), round(percent, 2)


    def predict(self, image_path):
        # YOLO 추론 실행
        result = self.model.predict(
            source=image_path, conf=0.5, show=False, show_boxes=False, save=False
        )[0]

        img_h, img_w = result.orig_shape
        total_pixels = img_h * img_w  # 전체 픽셀 수 (1920x1080)

        # 비율 및 카운트 초기화
        ratios = {"plastic": 0.0, "vinyl": 0.0, "wood": 0.0}
        real_areas = {"plastic": 0.0, "vinyl": 0.0, "wood": 0.0}
        count = 0

        # 원본 이미지 로드
        image = cv2.imread(image_path)

        if result.masks is not None:
            for mask, box, cls_id in zip(result.masks.data, result.boxes.xyxy, result.boxes.cls):
                mask_np = mask.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]))

                class_name = self.names[int(cls_id)]
                color = CLASS_COLORS.get(int(cls_id), (255, 255, 255))

                # 🔷 픽셀 기반 면적 계산 호출
                object_pixels = np.sum(mask_resized > 0.5)
                real_area, percent = self._calculate_object_area(object_pixels, total_pixels)

                # 통합 저장
                if class_name in ratios:
                    ratios[class_name] += percent
                    real_areas[class_name] += real_area

                # # 시각화
                # mask_img = np.zeros_like(image, dtype=np.uint8)
                # mask_img[mask_resized > 0.5] = color
                # image = cv2.addWeighted(mask_img, ALPHA, image, 1 - ALPHA, 0)

                # 시각화 (밝기 어두웠던 문제 해결)
                overlay = image.copy()
                overlay[mask_resized > 0.5] = color
                image = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)

                # x1, y1, x2, y2 = map(int, box)
                # cv2.putText(image, f"{class_name}", (x1, y1 - 5),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                # ===== 라벨 크기 키우기
                # === 라벨 그리기 (크게 + 가독성 높임) ===
                # 라벨 표시 (객체 색 그대로, 크기만 확대)
                x1, y1, x2, y2 = map(int, box)

                # 객체 크기에 비례한 글씨 크기 자동 조정 (너무 작을 때 대비)
                font_scale = max(0.8, min(2.5, (y2 - y1) / 80.0))
                thickness = max(2, int(font_scale * 2))

                label = f"{class_name}"
                cv2.putText(image, label, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 10)
                # =====

                count += 1

        # 결과 이미지 저장
        os.makedirs("temp", exist_ok=True)
        result_image_path = f"temp/result_{os.path.basename(image_path)}"
        cv2.imwrite(result_image_path, image)

        # 결과 이미지를 Base64로 변환
        with open(result_image_path, "rb") as f:
            result_img_base64 = base64.b64encode(f.read()).decode("utf-8")

        # 입력된 원본 이미지도 Base64 변환
        with open(image_path, "rb") as f:
            orig_img_base64 = base64.b64encode(f.read()).decode("utf-8")


        # 결과 딕셔너리 반환
        return {
            "orig_img": orig_img_base64,      # Base64 원본 이미지
            "rcnn_result": result_img_base64, # Base64 결과 이미지
            "wood": round(ratios["wood"], 2),
            "plastic": round(ratios["plastic"], 2),
            "vinyl": round(ratios["vinyl"], 2),
            "count": count
        }


# 검출 면적 계산 근거
# 촬영 높이(H), 화각(HFOV/VFOV)이 고정이라면 → 바닥 면적은 "항상 동일"한 고정값
# 즉, 95.6m²는 고정된 상수 값으로 간주할 수 있고, 코드에서 매번 계산할 필요가 없습니다.
# 바닥면적 S = 2 * H * tan(HFOV / 2) * 2 * H * tan(VFOV / 2) = 95.6m²
# “1920×1080 해상도는 바닥 면적 계산에 직접적으로 쓰이지 않는다”는 점을 이해하는 것이 핵심입니다.
# 해상도는 “바닥 면적이 아닌 바닥 면적을 나누는 분해능”


# 개념 정리
# 촬영 높이 + 화각 : 카메라가 실제로 볼 수 있는 공간의 크기	→ 전체 바닥 면적(m²)을 결정
# 1920×1080 해상도 : 그 공간을 몇 개의 픽셀로 나눠서 표현하는지	→ 픽셀당 실제 면적 계산에 사용


# 계산 흐름
# [실제 바닥 전체 면적 95.6m²]
#      ↓ 해상도 1920×1080에 의해 픽셀로 분할
# [2,073,600 픽셀]
#      ↓ YOLO가 마스크 처리로 특정 객체 영역 픽셀 수 계산
# [예: object_pixels = 200,000]
#      ↓ 픽셀 비율
# pixel_ratio = 200,000 / 2,073,600 ≈ 0.096
#      ↓ 실제 면적 계산
# object_real_area = 95.6m² * 0.096 ≈ 9.18m²


# 실행 테스트용
if __name__ == "__main__":
    weight_path = "../weights/best.pt"
    image_path = "C:/Users/301/Desktop/create_image/image_4.png"

    yolo = YOLOWrapper(weight_path)
    result = yolo.predict(image_path)

    print("\n========== 🔍 분석 결과 ==========")
    print(f"[YOLO] 총 감지 객체 수: {result['count']}")
    print(f"[YOLO] 나무 비율: {result['wood']}%")
    print(f"[YOLO] 플라스틱 비율: {result['plastic']}%")
    print(f"[YOLO] 비닐 비율: {result['vinyl']}%")
    print("==================================\n")

    # ✅ Base64 → OpenCV 이미지로 변환
    decoded = base64.b64decode(result['rcnn_result'])
    np_img = np.frombuffer(decoded, np.uint8)
    final_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    cv2.imshow("YOLO Detection Result", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()