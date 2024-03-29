import math
import copy
import cv2 as cv
import HandTrackingModule as htm

wcam, hcam = 960, 540

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc('M', 'J', 'P', 'G'))
cap.set(3, wcam)
cap.set(4, hcam)

tipIds = [8, 12, 16, 20]
count = 0
webcam = True

detector = htm.handDetector(maxHands=2, detectionCon=0.7)
cvFpsCalc = htm.CvFpsCalc(buffer_len=10)

def getCount(lmList):
    fingers = []
    pinkyKnuckle = lmList[17]
    thumbTip = lmList[4]
    thumbJoint = lmList[3]
    wrist = lmList[0]

    # Count thumb
    x1, y1 = pinkyKnuckle[1:]
    x2, y2 = thumbJoint[1:]
    x3, y3 = thumbTip[1:]
    if math.sqrt((x3 - x1)**2 + (y3 - y1)**2) > math.sqrt((x2 - x1)**2 + (y2 - y1)**2):
        fingers.append(1)
    else:
        fingers.append(0)

    # Count rest 4 fingers
    for tip in tipIds:
        a1, b1 = wrist[1:]
        a2, b2 = lmList[tip-2][1:]
        a3, b3 = lmList[tip][1:]
        if math.sqrt((a3 - a1)**2 + (b3 - b1)**2) > math.sqrt((a2 - a1)**2 + (b2 - b1)**2):
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

while True:
    fingers = []
    count = 0
    success, image = cap.read()
    if not success :
        break

    if webcam :
        image = cv.flip(image, 1)
    debug_image = copy.deepcopy(image) # Unaltered copy of image

    detector.findHands(debug_image)
    detector.drawHands(image)
    lmList = detector.getlmList(debug_image)
    handedness = detector.getHandedness()


    if len(lmList) != 0:
        for index, _  in enumerate(handedness):
            fingers.append(getCount(lmList[index]))
            count+=fingers[index].count(1)

    # Draw count
    cv.putText(image, f"Count: {count}", (10, 55), cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 7)
    cv.putText(image, f"Count: {count}", (10, 55), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)

    # Draw fps
    fps = cvFpsCalc.get()
    cv.putText(image, f"fps: {int(fps):02}", (835, 525), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    cv.putText(image, f"fps: {int(fps):02}", (835, 525), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

    if not webcam :
        image = cv.flip(image, 1)
    cv.imshow("image", image)
    key = cv.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv.destroyAllWindows()
