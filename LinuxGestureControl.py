import cv2 as cv
import time
import HandTrackingModule as htm
import pyautogui as pag

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

    gesture = detector.findGesture(img)
    img = detector.findHands(img)
    # if gesture == 
    print(gesture)

    cv.putText(img, str(int(fps)), (10, 60), 
               cv.FONT_HERSHEY_COMPLEX, 2, (25, 25, 200), 3)

    cv.imshow("Img", img)
    cv.waitKey(1)