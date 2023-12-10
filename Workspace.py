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
# def reset():
#     R, L, U, D = False, False, False, False


while True:
    success, img = cap.read()

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    gesture = detector.findGesture(img)
    img = detector.findHands(img)

    cv.putText(img, str(int(fps)), (10, 60), 
               cv.FONT_HERSHEY_COMPLEX, 2, (25, 25, 200), 3)

    cv.imshow("Img", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    lmList = detector.findPosition(img)

    mf = []
    if len(lmList) != 0:
        mf = lmList[12][1:3] 
        print(mf)        
    else:
        # R, L, D, U = False, False, False, False
    #     primer = False
        continue


    if primer == False:
        if R and mf[0] < 250:
            pag.hotkey('right')
            U, D, R = False, False, False
            continue
        elif L and mf[0] > 550:
            pag.hotkey('left')
            U, D, L = False, False, False
            continue
        if mf[0] > 550 and 150 < mf[1] < 450:
            R = True
            continue
        elif mf[0] < 250 and 150 < mf[1] < 450:
            L = True
            continue
    
    if workOpen:
        if gesture == 'Closed_Fist':
            primer = True
            R, L, D, U = False, False, False, False
            continue
        if primer and gesture == 'Open_Palm':
            pag.hotkey('enter')
            primer = False
            workOpen = False
            
        if D and mf[1] > 450:
            pag.hotkey('down')
            R, L, D = False, False, False
            continue
        elif U and mf[1] < 150:
            pag.hotkey('up')
            R, L, U = False, False, False
            continue
        elif mf[1] < 150 and 250 < mf[0] < 550:
            D = True
            continue
        elif mf[1] > 450 and 250 < mf[0] < 550:
            U = True
            continue
    else:
        if gesture == "Closed_Fist":
            primer = True
            continue
        if primer and gesture == "Open_Palm":
            pag.hotkey('ctrl', 'alt', 'up')
            primer = False
            workOpen = True

    # print(gesture)