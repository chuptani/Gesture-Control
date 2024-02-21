import cv2 as cv
import time
import numpy as np
import HandTrackingModule as htm
import math as m

###############################
wCam, hCam = 800, 600
##############################

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(maxHands=1, detectionCon=0.7)

while True:
    success, img = cap.read()

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        cv.circle(img, (x1, y1), 15, (200, 40, 25), cv.FILLED)
        cv.circle(img, (x2, y2), 15, (200, 40, 25), cv.FILLED)
        cv.line(img, (x1, y1), (x2, y2), (200, 40, 25), 3)
        cv.circle(img, (cx, cy), 15, (200, 40, 25), cv.FILLED)

        length = m.hypot(x2 - x1, y2 - y1)
        # print(length)
        if length < 50:
            cv.circle(img, (cx, cy), 15, (25, 25, 200), cv.FILLED)


    cv.putText(img, str(int(fps)), (10, 60), 
               cv.FONT_HERSHEY_COMPLEX, 2, (25, 25, 200), 3)

    cv.imshow("Img", img)
    cv.waitKey(1)