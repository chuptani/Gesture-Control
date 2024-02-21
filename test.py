import copy
import cv2 as cv
import HandTrackingModule as htm

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc('M', 'J', 'P', 'G'))
cap.set(3, 960)
cap.set(4, 540)

pTime = 0

detector = htm.handDetector()
cvFpsCalc = htm.CvFpsCalc(buffer_len=10)

while True:

    success, image = cap.read()
    if not success :
        break

    image = cv.flip(image, 1)
    debug_image = copy.deepcopy(image)

    detector.findHands(debug_image)
    detector.drawHands(image)

    fps = cvFpsCalc.get()
    cv.putText(image, str("{:02}".format(int(fps))), (865, 535), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3)

    cv.imshow("image", image)
    key = cv.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv.destroyAllWindows()
