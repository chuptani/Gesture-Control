import time
import cv2 as cv
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class handDetector:
    def __init__(
        self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5
    ):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplexity = modelComplexity
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands #type: ignore
        self.hands = self.mpHands.Hands(
            self.mode,
            self.maxHands,
            self.modelComplexity,
            self.detectionCon,
            self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils #type: ignore

        base_options = python.BaseOptions(model_asset_path="./gesture_recognizer.task")
        options = vision.GestureRecognizerOptions(base_options=base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(options)


    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        print(self.results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )

        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                height, width, _ = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)

        return lmList

    def findGesture(self, img):
        gesture = "none"
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        recognition_result = self.recognizer.recognize(image)
        if len(recognition_result.gestures) != 0:
            gesture = recognition_result.gestures[0][0].category_name

        return gesture


class CvFpsCalc(object):
    def __init__(self, buffer_len=1):
        self._start_tick = cv.getTickCount()
        self._freq = 1000.0 / cv.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded


def main():


    cap = cv.VideoCapture(0)

    detector = handDetector()
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:

        _, image = cap.read()
        image = cv.flip(image, 1)

        gesture = detector.findGesture(image)
        print(gesture)
        
        image = detector.findHands(image)

        lmList = detector.findPosition(image, draw=False)
        if len(lmList) != 0:
            print(lmList)

        fps = cvFpsCalc.get()

        cv.putText(
            image, str(int(fps)), (5, 30), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2
        )

        cv.imshow("Image", image)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break


if __name__ == "__main__":
    main()
