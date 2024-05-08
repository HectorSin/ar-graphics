import cv2

# 사용 가능한 카메라의 인덱스를 찾습니다.
index = 0
arr = []
while True:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.read()[0]:
        break
    else:
        arr.append(index)
    cap.release()
    index += 1

print("Available camera indexes:", arr)  # 사용 가능한 카메라 인덱스 출력
