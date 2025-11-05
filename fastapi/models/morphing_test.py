import cv2
import numpy as np


# ✅ 테스트 설정
#IMAGE_PATH = "C:/Users/dayeon/Desktop/create_image/ChatGPT_Image.png"
IMAGE_PATH = "C:/Users/301/Desktop/create_image/image_yolo31.png"
H, S, V = 70, 150, 90  # 원하는 HSV 값으로 변경
OUTPUT_PREFIX = 'morph_step_'


def save_step(image, step_no, name):
   filename = f"{OUTPUT_PREFIX}{step_no}_{name}.png"
   cv2.imwrite(filename, image)
   print(f"[저장 완료] {filename}")


def main():
   img = cv2.imread(IMAGE_PATH)
   img = cv2.resize(img, (400, 400))

   if img is None:
       print("❌ 이미지 로드 실패. 경로를 확인하세요.")
       return

   print("✅ 이미지 shape:", img.shape)


   # 1️⃣ 원본
   step1 = img.copy()
   cv2.imshow("step1", step1)
   #save_step(step1, 1, 'original')


   # 2️⃣ 그레이스케일
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   step2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
   cv2.imshow("step2", step2)
   #save_step(step2, 2, 'grayscale')


   # 3️⃣ Sobel Edge
   sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
   sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
   sobel_mag = cv2.convertScaleAbs(cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0))
   step3 = cv2.cvtColor(sobel_mag, cv2.COLOR_GRAY2BGR)
   cv2.imshow("step3", step3)
   #save_step(step3, 3, 'sobel_edge')


   # 4️⃣ Canny Edge
   edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 80, 160)
   step4 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
   cv2.imshow("step4", step4)
   #save_step(step4, 4, 'canny_edge')


   # 5️⃣ HSV 변환 및 마스크 생성
   # BGR to HSV 변환
   hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   lower = np.array([60, 120, 0])
   upper = np.array([H, S, V])

   # 색상 범위를 제한하여 mask 생성
   mask = cv2.inRange(hsv_img, lower, upper)


   # 마스크만 흑백 표시
   step5 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
   cv2.imshow("step5", step5)
   #save_step(step5, 5, 'mask_black_white')


   # 6️⃣ 컬러 강조된 마스크 (bitwise_and)

   # 원본 이미지를 가지고 Object 추출 이미지로 생성
   masked_color = cv2.bitwise_and(img, img, mask=mask)
   step6 = masked_color
   cv2.imshow("step6", step6)
   #save_step(step6, 6, 'mask_color_highlight')


   # 7️⃣ Otsu 이진화
   _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
   step7 = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
   cv2.imshow("step7", step7)
   #save_step(step7, 7, 'otsu_binary')


   # 8️⃣ Morphology (Opening)
   kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
   opened = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel, iterations=1)
   step8 = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
   cv2.imshow("step8", step8)
   #save_step(step8, 8, 'morph_open')

   print("🎉 모든 단계 이미지 생성 완료!")

   cv2.waitKey(0)
   cv2.destroyAllWindows()


if __name__ == "__main__":
   main()
