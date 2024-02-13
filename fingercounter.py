import cv2 as cv
import time
import HandTrackingModule as htm

# import os
# import numpy as np

wcam, hcam = 1280, 720

cap = cv.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)

pTime = 0

folderPath = "images/counting"
# myList = os.listdir(folderPath)
myList = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg"]
overlayList = []
new_shape = (720, 512, 3)
tipIds = [8, 12, 16, 20]
count = 0

for imPath in myList:
    image = cv.imread(f"{folderPath}/{imPath}")
    overlayList.append(image)

detector = htm.handDetector(maxHands=1, detectionCon=0.75)
cvFpsCalc = htm.CvFpsCalc(buffer_len=10)

while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = []

        if lmList[4][1] < lmList[3][1]:
            fingers.append(0)
        else:
            fingers.append(1)

        for id in tipIds:
            if lmList[id][2] < lmList[id - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        count = fingers.count(1)

        img[0:720, 0:512] = overlayList[count - 1]

    fps = cvFpsCalc.get()

    cv.putText(
        img, str(int(fps)), (1180, 700), cv.FONT_HERSHEY_COMPLEX, 2, (214, 11, 299), 3
    )

    cv.imshow("Img", img)
    key = cv.waitKey(1)
    if key == 27:  # ESC
        break

    # if cv.waitKey(1) & 0xFF == ord("q"):
    #     break
