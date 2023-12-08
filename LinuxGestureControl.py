import cv2 as cv
import time
import HandTrackingModule as htm
import pyautogui as pag

##############################
wCam, hCam = 800, 600
##############################

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(maxHands=1, detectionCon=0.7)
primer = False
workOpen = False
ready = False
R, L, U, D = False, False, False, False
def reset():
    R, L, U, D = False, False, False, False


while True:
    success, img = cap.read()

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    gesture = detector.findGesture(img)
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        print(lmList[12])
        mf_x = lmList[12][1] 
        mf_y = lmList[12][2]
    
    if workOpen:
        if primer and gesture == 'Open_Palm':
            pag.hotkey('enter')
            primer = False
            workOpen = False
        elif gesture == 'Closed_Fist':
            primer = True
        if mf_x > 550:
            R = True
        elif mf_x < 250:
            L = True
        if mf_y < 150:
            D = True
        elif mf_y > 450:
            U = True
        if L or R or D or U:
            if L and mf_x > 550:
                pag.hotkey('left')
                L, R, D, U = 0, 0, 0, 0
            elif R and mf_x < 250:
                pag.hotkey('right')
                L, R, D, U = 0, 0, 0, 0
            elif D and mf_y > 450:
                pag.hotkey('down')
                L, R, D, U = 0, 0, 0, 0
            elif U and mf_y < 150:
                pag.hotkey('up')
                L, R, D, U = 0, 0, 0, 0

    else:
        if gesture == "Closed_Fist":
            primer = True
        if primer and gesture == "Open_Palm":
            pag.hotkey('ctrl', 'alt', 'up')
            primer = False
            workOpen = True

        

    # print(gesture)

    cv.putText(img, str(int(fps)), (10, 60), 
               cv.FONT_HERSHEY_COMPLEX, 2, (25, 25, 200), 3)

    cv.imshow("Img", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break