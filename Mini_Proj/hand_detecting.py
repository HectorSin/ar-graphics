import cv2
from cvzone.HandTrackingModule import HandDetector 
import handDetector as htm

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
#detector = HandDetector(detectionCon=0.8, maxHands=2)
detector = htm.handDetector(detectionCon=0.75)
colorR = (255, 0, 255)

cx, cy, w, h = 100, 100, 200, 200

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame from video capture. Check your camera connection.")
        break
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    # hands, img = detector.findHands(img)  # with draw
    # hands = detector.findHands(img, draw=False) # without draw

    """
    lmList, _ = detector.findPosition(img)

    if lmList:
        l, _, _ = detector.findDistance(8, 12, img, draw=False)
        print(l)
        if l < 30:
            cursor = lmList[8] # index finger tip landmark
            # call the update here
    """

    out = img.copy()
    cv2.imshow("Image", out)
    cv2.waitKey(1)