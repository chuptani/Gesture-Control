import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, 
                 modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplexity = modelComplexity
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, 
                                        self.maxHands, 
                                        self.modelComplexity,
                                        self.detectionCon,
                                        self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
        base_options = python.BaseOptions(model_asset_path='./gesture_recognizer.task')
        options = vision.GestureRecognizerOptions(base_options=base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(options) 
        
    def findHands(self, img, draw=True):
        
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, 
                                               self.mpHands.HAND_CONNECTIONS)


        return img 

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape 
                cx, cy = int(lm.x*w), int(lm.y*h) 
                lmList.append([id, cx, cy]) 
                # if draw: 
                #     cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED) 

        return lmList   

    def findGesture(self, img):

        gesture = "none"
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                         data=imgRGB)
        recognition_result = self.recognizer.recognize(image)
        if len(recognition_result.gestures) != 0:
            gesture = recognition_result.gestures[0][0].category_name

        return gesture

def main():
    pTime = 0 
    cTime = 0
    cap = cv.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        gesture = detector.findGesture(img)
        print(gesture)
        img = detector.findHands(img)

        # lmList = detector.findPosition(img, draw=False)
        # if len(lmList) != 0:
        #     print(lmList[4])

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (5,30), 
                   cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

        cv.imshow('Image', img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
