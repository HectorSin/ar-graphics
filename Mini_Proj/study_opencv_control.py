import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

"""
for i in range(5):  # try indices from 0 to 4
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        break
else:
    print("No camera found")
    exit(1)
"""

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=80, maxHands=2)
colorR = (255, 0, 255)

cx, cy, w, h = 100, 100, 200, 200


class DragRect():
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter
        self.size = size

    def update(self, cursor):
        cx, cy = self.posCenter
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
    """
    lmList, _ = detector.findPosition(img)
    

    if lmList:

        l, _, _ = detector.findDistance(8, 12, img, draw=False)
        print(l)
        if l < 30:
            cursor = lmList[8]  # index finger tip landmark
            # call the update here
            for rect in rectList:
                rect.update(cursor)
"""    
    ## Draw solid
    # for rect in rectList:
    #     cx, cy = rect.posCenter
    #     w, h = rect.size
    #     cv2.rectangle(img, (cx - w // 2, cy - h // 2),
    #                   (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
    #     cvzone.cornerRect(img, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    ## Draw Transperency
    """
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2),
                      (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)
"""
    out = img.copy()
    alpha = 0.5
    #mask = imgNew.astype(bool)
    #out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow("Image", out)
    cv2.waitKey(1)