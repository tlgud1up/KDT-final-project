import os
import numpy as np
import joblib
import cv2
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO
from sklearn.decomposition import PCA

# https://doing-nothing.tistory.com/79 pca 분석 정규화 과정 참고

# 본 코드는 YOLO 세그멘테이션 결과(마스크 픽셀 수)를 이미지 전체 픽셀 수로 나눠
# 클래스별 '면적 비율(feature)'을 만든 뒤, 이를 표준화 → PCA(2차원)로 축소하고
# 모델(Scaler, PCA)과 중간 결과(features 등)을 저장합니다.

# -------------------------------
# 1. YOLO 모델 로드
# -------------------------------
# 학습된 세그멘테이션 가중치(.pt) 경로를 지정하여 모델을 메모리에 적재
model = YOLO("../weights/best.pt")

# 클래스 이름 정의
# YOLO 결과(result.boxes.cls)는 정수 인덱스로 나오므로, 이름으로 변환
CLASS_NAMES = {0: 'wood', 1: 'vinyl', 2: 'plastic'}
CLASS_LIST = ['wood', 'vinyl', 'plastic']

# -------------------------------
# 2. YOLO 특징 추출 함수
# -------------------------------
def extract_ratios(image_path):
    """
        단일 이미지에 대해 YOLO 세그멘테이션을 수행하고,
        각 클래스(wood, vinyl, plastic)가 차지하는 '마스크 픽셀 수 / 전체 픽셀 수' 비율을 반환
        반환: [wood_ratio, vinyl_ratio, plastic_ratio] (각 0~1 실수)
        실패 시 None 반환
    """
    # YOLO 추론 (Ultralytics는 경로 문자열을 바로 넣어도 내부에서 읽어 처리 가능)
    results = model(image_path)

    # OpenCV로 원본 이미지를 읽어 크기(픽셀 수)를 구함
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 이미지 로드 실패: {image_path}")
        return None

    h, w, _ = img.shape
    total_area = h * w  # 전체 픽셀 수

    # 클래스별 누적 면적(픽셀 수) 초기화
    class_areas = {name: 0 for name in CLASS_LIST}

    for result in results:
        masks = result.masks    # 세그멘테이션 마스크 객체
        if masks is None:       # 해당 이미지에서 세그멘테이션이 1개도 안 잡힌 경우
            continue

        # masks.data: shape = [N, H, W], 값은 0/1(또는 0.0/1.0) float 텐서
        # result.boxes.cls: 길이 N, 각 인스턴스의 클래스 인덱스
        for mask, cls in zip(masks.data, result.boxes.cls):
            # 디버깅용 출력: 마스크 텐서 크기와 클래스 인덱스
            print("마스크 텐서 크기와 클래스 인덱스")
            print(mask.shape, cls)

            # 클래스 인덱스를 사람이 읽기 쉬운 이름으로 변환
            cls_name = CLASS_NAMES[int(cls)]

            # 텐서 → NumPy → uint8(0/1)로 변환 후 픽셀 합산
            mask_np = mask.cpu().numpy().astype(np.uint8)
            class_areas[cls_name] += np.sum(mask_np)

    # 각 클래스 면적을 전체 면적(total_area)로 나눠 비율로 변환
    return [
        class_areas['wood'] / total_area,
        class_areas['vinyl'] / total_area,
        class_areas['plastic'] / total_area
    ]

# -------------------------------
# 3. 기준 데이터셋 로드 및 특징 추출
# -------------------------------
dataset_dir = "C:/Users/301/Desktop/pca_image"
features = []
labels = []

for cls_index, cls_name in CLASS_NAMES.items():
    class_folder = os.path.join(dataset_dir, cls_name)

    # 폴더가 없을 때를 대비한 안전장치 (개선①)
    if not os.path.isdir(class_folder):
        print(f"⚠️ 경고: 폴더가 없습니다: {class_folder}")
        continue

    for filename in os.listdir(class_folder):
        if filename.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(class_folder, filename)
            ratios = extract_ratios(img_path)

            if ratios is not None:
                features.append(ratios)
                labels.append(cls_name)

features = np.array(features)   # shape: [num_samples, 3]
labels = np.array(labels)       # shape: [num_samples]

print("✅ 특징 추출 완료:", features.shape)

# -------------------------------
# 4. 표준화 + PCA 학습
# -------------------------------
# PCA는 변수 스케일의 영향을 많이 받습니다.
# 클래스별 면적 비율이라도 분포가 다를 수 있으므로, 평균0/표준편차1로 표준화 후 PCA를 학습합니다.
scaler = StandardScaler()
features_std = scaler.fit_transform(features)   # Z-정규화(표준화) 적용

# 주성분 2개(2차원)로 축소
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_std)

# 각 주성분이 설명하는 분산 비율 (합계는 1.0에 근사)
explained = pca.explained_variance_ratio_
print(f"📌 PCA 설명된 분산 비율 (Standardized): {explained}")

# -------------------------------
# 5. 저장
# -------------------------------
os.makedirs("pca", exist_ok=True)

joblib.dump(pca, "pca/pca_model.pkl")
joblib.dump(scaler, "pca/scaler.pkl")    # 표준화 스케일러

# 분석 중간 산출물 저장(선택적이지만 재현성/디버깅에 유용)
np.save("pca/features.npy", features)            # 원본 비율 특징(표준화 전)
np.save("pca/features_std.npy", features_std)    # 표준화된 특징
np.save("pca/labels.npy", labels)                # 레이블
np.save("pca/pca_explained.npy", explained)      # 설명된 분산비율

print("✅ 표준화 + PCA 모델 저장 완료!")