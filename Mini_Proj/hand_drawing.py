import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 손 인식 초기화
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# OpenCV 윈도우 초기화
cap = cv2.VideoCapture(0)
canvas = None
prev_x, prev_y = 0, 0  # 이전 좌표 초기화

# 그림을 그릴 때의 최대 거리 설정
DRAWING_DISTANCE = 0.03  # 임계값은 실험을 통해 조정 가능

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # 이미지 처리
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 캔버스 생성
        if canvas is None:
            canvas = np.zeros_like(image)

        # 손 검출
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 엄지와 검지의 끝 좌표 구하기
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                x1, y1 = int(index_finger_tip.x * image.shape[1]), int(index_finger_tip.y * image.shape[0])

                # 거리 계산
                distance = np.linalg.norm([index_finger_tip.x - thumb_tip.x, index_finger_tip.y - thumb_tip.y])

                # 거리에 따라 그림 그리기
                if distance < DRAWING_DISTANCE:
                    if prev_x != 0 and prev_y != 0:
                        cv2.line(canvas, (prev_x, prev_y), (x1, y1), [255, 0, 0], 4)
                    prev_x, prev_y = x1, y1  # 현재 좌표 업데이트
                else:
                    prev_x, prev_y = 0, 0  # 거리가 멀어지면 좌표 초기화

        # 캔버스를 이미지에 겹치기
        combined = cv2.addWeighted(image, 0.5, canvas, 0.5, 0)
        cv2.imshow('Hand Tracking', combined)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
