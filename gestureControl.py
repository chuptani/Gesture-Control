import csv
import math
import copy
import argparse
import subprocess

import cv2 as cv
import HandTrackingModule as htm

from collections import deque, Counter
from model import PointHistoryClassifier

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--train", action="store_true")
    return parser.parse_args()


def main():

    args = getArgs()

    wcam, hcam = 960, 540

    cap = cv.VideoCapture(args.device)
    # My laptop is annoying
    if args.device == 0:
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap.set(3, wcam)
    cap.set(4, hcam)

    detector = htm.handDetector(maxHands=1, detectionCon=0.5)
    cvFpsCalc = htm.CvFpsCalc(buffer_len=10)

    pointHistoryClassifier = PointHistoryClassifier()

    with open("model/point_history_classifier/point_history_classifier_label.csv",encoding="utf-8-sig") as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [row[0] for row in point_history_classifier_labels]


    countHistory = deque([0] * 5, maxlen=5)
    number = 0
    code = []
    ready = True
    zeroCount = 0
    warningGiven = False
    webcam = True
    pointHistoryLength = 16
    pointHistory = deque(maxlen=pointHistoryLength)
    fingerGestureIdHistory =[]
    startMode = 2 if args.train else 0
    mode = startMode
    print(startMode)

    while True:
        fingers = []
        count = 0
        label = -1

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        elif 48 <= key <= 57:  # 0 ~ 9
            label = key - 48
        elif key == 113: # q
            label = -1

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

        if warningGiven == True and len(lmList) != 0:
            print("\033[38;5;34mHand detected\033[0m")
            mode = startMode
            warningGiven = False
        elif warningGiven == False and len(lmList) == 0:
            print("\033[38;5;160mWARNING: No hand detected\033[0m")
            mode=9
            code = []
            fingerGestureIdHistory = []
            warningGiven = True

        match mode:
            case 0:
                for index, _  in enumerate(handedness):
                    fingers.append(getCount(lmList[index]))
                    count+=fingers[index].count(1)
                countHistory.append(count)
                countCounter = Counter(countHistory)
                number = countCounter.most_common(1)[0][0]

                if zeroCount == 15 and len(code) != 0:
                    zeroCount+=1
                    code = []
                    print("\033[38;5;34mRESET\033[0m")
                elif zeroCount <= 15 and number == 0:
                    zeroCount+=1
                    ready = True
                elif len(code) == 2:
                    mode, exitCode = run(code)
                    if exitCode == 1:
                        break
                    code = []
                    if mode == 0:
                        print("\033[38;5;34mRESET\033[0m")
                    elif mode == 1:
                        print("Reading gesture...")
                elif ready and number != 0:
                    zeroCount = 0
                    code.append(number)
                    print(code)
                    ready = False
            case 1:
                pointHistory.append(lmList[0][8][1:3])
                processedPointHistory = processPointHistory(debug_image, pointHistory)
                if len(processedPointHistory) == pointHistoryLength * 2:
                    tempFingureGestureId = pointHistoryClassifier(processedPointHistory)
                    fingerGestureIdHistory.append(tempFingureGestureId)
                    # print(tempFingureGestureId)
                if len(fingerGestureIdHistory) == 40:
                    fingerGestureID = Counter(fingerGestureIdHistory).most_common(1)[0][0]
                    gesture = point_history_classifier_labels[fingerGestureID]
                    print(gesture)
                    fingerGestureIdHistory = []
                    mode = 0
                    print("\033[38;5;34mRESET\033[0m")
            case 2:
                pointHistory.append(lmList[0][8][1:3])
                processedPointHistory = processPointHistory(debug_image, pointHistory)
                if len(processedPointHistory) == pointHistoryLength * 2:
                    csv_path = "model/point_history_classifier/point_history.csv"
                    with open(csv_path, "a", newline="") as file:
                        writer = csv.writer(file)
                        if label != -1:
                            writer.writerow([label, *processedPointHistory])

        if startMode == 2:
            cv.putText(image, f"Training Mode", (710, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            cv.putText(image, f"Training Mode", (710, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

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

    cap.release()
    cv.destroyAllWindows()


def getCount(lmList):
    fingers = []
    tipIds = [8, 12, 16, 20]
    pinkyKnuckle = lmList[17]
    thumbTip = lmList[4]
    thumbJoint = lmList[3]
    wrist = lmList[0]

    # Count thumb
    # TODO: fix thumb outside fist counting problem
    x1, y1 = pinkyKnuckle[1:3]
    x2, y2 = thumbJoint[1:3]
    x3, y3 = thumbTip[1:3]
    if math.sqrt((x3 - x1)**2 + (y3 - y1)**2) > math.sqrt((x2 - x1)**2 + (y2 - y1)**2):
        fingers.append(1)
    else:
        fingers.append(0)

    # Count rest 4 fingers
    for tip in tipIds:
        a1, b1 = wrist[1:3]
        a2, b2 = lmList[tip-2][1:3]
        a3, b3 = lmList[tip][1:3]
        if math.sqrt((a3 - a1)**2 + (b3 - b1)**2) > math.sqrt((a2 - a1)**2 + (b2 - b1)**2):
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


def run(code):
    mode = 0
    exitCode = 0
    if code[1] == 5 and 1 <= code[0] <= 9:
        subprocess.run(f'xdotool key super+{str(code[0])}', shell=True)
    elif code == [2, 2]:
        subprocess.run(f'st &', shell=True)
    elif code == [3, 3]:
        subprocess.run(f'xdotool key super+q', shell=True)
    elif code == [1,1]:
        mode=1
    elif code == [1, 3]:
        exitCode = 1
    else:
        print("\033[38;5;160mUnknown command\033[0m")

    return mode, exitCode


def processPointHistory(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    # base_z = 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]
            # base_z = point[2]

        temp_point_history[index][0] = (
            temp_point_history[index][0] - base_x
        ) / image_width
        temp_point_history[index][1] = (
            temp_point_history[index][1] - base_y
        ) / image_height
        # temp_point_history[index][2] = (
        #     temp_point_history[index][1] - base_z
        # ) / 1000

    # Convert to a one-dimensional list
    temp_point_history = [coordinate for point in temp_point_history for coordinate in point]
    return temp_point_history


if __name__ == "__main__":
    main()
