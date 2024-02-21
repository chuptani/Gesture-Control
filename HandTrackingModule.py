import copy
import cv2 as cv
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class handDetector:
    def __init__(
        self,
        mode=False,
        maxHands=2,
        modelComplexity=1,
        detectionCon=0.5,
        trackCon=0.5
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


    def findHands(self, image):
        process_image = cv.cvtColor(image, cv.COLOR_BGR2RGB) # Convert to RGB
        image.flags.writeable = False
        self.results = self.hands.process(process_image)
        image.flags.writeable = True


    def drawHands(self, image):
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)

        return image


    def getHandedness(self):
        handedness = []
        if self.results.multi_handedness:
            for hand in self.results.multi_handedness:
                handedness.append(hand.classification[0].label)

        return handedness


    def getlmList(self, image):
        lmList = []
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                landmarkList = []
                for id, lm in enumerate(hand.landmark):
                    height, width, _ = image.shape
                    lmx, lmy = int(lm.x * width), int(lm.y * height)
                    landmarkList.append([id, lmx, lmy])
                lmList.append(landmarkList)

        return lmList

    def getGesture(self, img):
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
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    detector = handDetector()
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        _, image = cap.read()
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        detector.findHands(debug_image)
        detector.drawHands(image)
        lmList = detector.getlmList(debug_image)
        gesture = detector.getGesture(debug_image)

        if len(lmList) != 0:
            for lms in lmList:
                print(lmList[lms])

        print(gesture)

        fps = cvFpsCalc.get()
        cv.putText(image, str(int(fps)), (5, 30), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
        cv.imshow("Image", image)
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break


if __name__ == "__main__":
    main()
