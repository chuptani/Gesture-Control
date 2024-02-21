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
    indexKnuckle = lmList[5]
    pinkyKnuckle = lmList[17]
    thumbTip = lmList[4]
    thumbJoint = lmList[3]
    wrist = lmList[0]
    x = 1
    y = 2

    # def thumbCount():
    # x1, y1 = pinkyKnuckle[1:]
    # x2, y2 = indexKnuckle[1:]
    # x3, y3 = thumbTip[1:]
    #
    # if math.sqrt((x2 - x1)**2 + (y2 - y1)**2) < math.sqrt((x3 - x1)**2 + (y3 - y1)**2):
    #     fingers.append(1)
    # else:
    #     fingers.append(0)

    def thumbVertical():
        if indexKnuckle[x] < pinkyKnuckle[x]:
            if thumbTip[x] < thumbJoint[x]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if thumbTip[x] > thumbJoint[x]:
                fingers.append(1)
            else:
                fingers.append(0)

    def thumbHorizontal():
        if indexKnuckle[y] < pinkyKnuckle[y]:
            if thumbTip[y] < thumbJoint[y]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if thumbTip[y] > thumbJoint[y]:
                fingers.append(1)
            else:
                fingers.append(0)

    if indexKnuckle[y] < wrist[y] and pinkyKnuckle[y] < wrist[y]:
        thumbVertical()
        for id in tipIds:
            if lmList[id][y] < lmList[id - 2][y]:
                fingers.append(1)
            else:
                fingers.append(0)
    elif indexKnuckle[y] > wrist[y] and pinkyKnuckle[y] > wrist[y]:
        thumbVertical()
        for id in tipIds:
            if lmList[id][y] > lmList[id - 2][y]:
                fingers.append(1)
            else:
                fingers.append(0)
    elif indexKnuckle[x] < wrist[x] and pinkyKnuckle[x] < wrist[x]:
        thumbHorizontal()
        for id in tipIds:
            if lmList[id][x] < lmList[id - 2][x]:
                fingers.append(1)
            else:
                fingers.append(0)
    elif indexKnuckle[x] > wrist[x] and pinkyKnuckle[x] > wrist[x]:
        thumbHorizontal()
        for id in tipIds:
            if lmList[id][x] > lmList[id - 2][x]:
                fingers.append(1)
            else:
                fingers.append(0)

    return fingers.count(1)

while True:
    count = 0
    success, image = cap.read()
    if not success :
        break

    if webcam :
        image = cv.flip(image, 1)
    debug_image = copy.deepcopy(image) # unaltered copy of image

    detector.findHands(debug_image)
    detector.drawHands(image)
    lmList = detector.getlmList(debug_image)
    handedness = detector.getHandedness()


    if len(lmList) != 0:
        for index, _  in enumerate(handedness):
            count+=getCount(lmList[index])

    cv.putText(image, f"Count: {count}", (10, 55), cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 7)
    cv.putText(image, f"Count: {count}", (10, 55), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)

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

