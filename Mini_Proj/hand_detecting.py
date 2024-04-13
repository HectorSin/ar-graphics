import cv2
from cvzone.HandTrackingModule import HandDetector 
import handDetector as htm
import cvzone
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8, maxHands=2)
#detector = htm.handDetector(detectionCon=0.75)
colorR = (255, 0, 255)

# print(dir(detector))

cx, cy, w, h = 100, 100, 200, 200

class DragRect():
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter
        self.size = size

    def update(self, cursor):
        posCenter = self.posCenter
        if len(posCenter) == 2:
            cx, cy = posCenter
            w, h = self.size

            # If the index finger tip is in the rectangle region
            if cx - w // 2 < cursor[0] < cx + w // 2 and \
                    cy - h // 2 < cursor[1] < cy + h // 2:
                self.posCenter = cursor


rectList = []
for x in range(5):
    rectList.append(DragRect([x * 250 + 150, 150]))

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame from video capture. Check your camera connection.")
        break
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)  # with draw
    # hands = detector.findHands(img, draw=False) # without draw

    if hands:
        # Information for the first hand detected
        hand1 = hands[0]  # Get the first hand detected
        lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand
        bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)
        center1 = hand1['center']  # Center coordinates of the first hand
        handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")

        # Count the number of fingers up for the first hand
        fingers1 = detector.fingersUp(hand1)
        # print(f'H1 = {fingers1.count(1)}', end=" ")  # Print the count of fingers that are up

        # Calculate distance between specific landmarks on the first hand and draw it on the image
        length, info, img = detector.findDistance(lmList1[4][0:2], lmList1[8][0:2], img, color=(255, 0, 255),
                                                  scale=10)

        # print(length, info)  # Print the length and the information about the distance

        if length < 30:
            cursor = lmList1[8]  # index finger tip landmark
            # call the update here
            for rect in rectList:
                rect.update(cursor)    

    imgNew = np.zeros_like(img, np.uint8)
    
    for rect in rectList:
        posCenter = rect.posCenter
        if len(posCenter) == 2:
            cx, cy = posCenter
            w, h = rect.size
            cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2),
                        (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
            cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    out = img.copy()

    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow("Image", out)
    cv2.waitKey(1)