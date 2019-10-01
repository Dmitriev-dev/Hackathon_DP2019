import imutils
import time
import dlib
import cv2
from imutils import face_utils
from scipy.spatial import distance as dist
import math
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import webbrowser
import requests
import pygame



class Worker():

    def __init__(self):
        self.eyes = None
        self.defects = []
        self.EYE_AR_THRESH = 0.23
        self.EYE_AR_CONSEC_FRAMES = 3
        self.COUNTER = 0
        self.TOTAL = 0
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("models/face_landmarks.dat")
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.rotate_str = "Normal"
        self.cap = None
        self.detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
        self.emotion_model_path = 'models/emotions.hdf5'
        self.face_detection = cv2.CascadeClassifier(self.detection_model_path)
        self.emotion_classifier = load_model(self.emotion_model_path, compile=False)
        self.Emotion = ["angry", "disgust", "scared", "happy", "sad", "surprised",
                        "neutral"]
        self.emotion_probs = [0, 0, 0]
        self.hertz = 0 #усталость+
        self.h = 0#высота плеч+
        self.h_eyes = 0#ысота глаз+
        self.rotate_head = 0#наклон головы +
        self.iteration = 0
        self.rating = 0
        self.emotion_prob = [0, 0, 0]
        self.image = None



    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def form_rate(self):
        self.rating = int((self.rating + self.rotate_head + self.hertz \
                      + self.h - int((self.emotion_prob[0]  * 2 -
                                  self.emotion_prob[1] + self.emotion_prob[2]/2))) /20)

 #я вставил эту фигню
    def get_diff_position_eye_pupil(self, gray_image, left_eye) :
        max_pos_eye = left_eye.max(axis=0)
        min_pos_eye = left_eye.min(axis=0)
        shift = 3
        min_pos_eye -= shift
        max_pos_eye += shift
        eye_cut = gray_image[min_pos_eye[1]: max_pos_eye[1], min_pos_eye[0]: max_pos_eye[0]]
        eye_cut = cv2.equalizeHist(eye_cut)

       #cv2.namedWindow("cut", cv2.WINDOW_KEEPRATIO)
        #cv2.imshow("cut", eye_cut)

        threshold_value = 10
        max_binary_value = 60
        _, threshold = cv2.threshold(eye_cut, threshold_value, max_binary_value, cv2.THRESH_BINARY_INV)
        dilatation_size = 2
        dilatation_type = cv2.MORPH_ELLIPSE
        element = cv2.getStructuringElement(dilatation_type, (2 * dilatation_size + 1, 2 * dilatation_size + 1), (dilatation_size, dilatation_size))
        binary = cv2.dilate(threshold, element)

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # return error
        error_my = float(10e6)
        if not contours:
            return error_my, error_my
        pupil = contours[0]
        # calculate moments for each contour
        m_pupil = cv2.moments(pupil)
        m_eye = cv2.moments(left_eye)
        # calculate x,y coordinate of center
        if m_pupil["m00"] != 0 and m_eye["m00"] != 0:
            center_x_pupil = float(m_pupil["m10"] / m_pupil["m00"])
            center_y_pupil = float(m_pupil["m01"] / m_pupil["m00"])

            center_x_eye = float(m_eye["m10"] / m_eye["m00"]) - float(min_pos_eye[0])
            center_y_eye = float(m_eye["m01"] / m_eye["m00"]) - float(min_pos_eye[1])
        else:
            center_x_pupil = error_my
            center_y_pupil = error_my

            center_x_eye = 2 * error_my
            center_y_eye = 2 * error_my

        cv2.circle(binary, (int(center_x_pupil), int(center_y_pupil)), 5, (255, 255, 255), -1)
        delta_x = center_x_eye - center_x_pupil
        delta_y = center_y_eye - center_y_pupil
        return delta_x, delta_y
        pass

    def get_number_of_quadrant(self, diff_pupil_pos_x, diff_pupil_pos_y):
        print(diff_pupil_pos_x)
        print(diff_pupil_pos_y)
        x_sum = sum(diff_pupil_pos_x)
        y_sum = sum(diff_pupil_pos_y)
        #print("x_sum = ", x_sum, ", y_sum = ", y_sum)
        name = "current"
        cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
        step_x = 1000;
        step_y = 1100;
        if x_sum > 0 and y_sum > 0:
            cv2.moveWindow(name, step_x, 1)
            #return 1
            print(1)
        if x_sum < 0 and y_sum > 0:
            cv2.moveWindow(name, 1, 1)
            #return 2
            print(2)
        if x_sum > 0 and y_sum < 0:
            cv2.moveWindow(name, 1, step_y)
            #return 3
            print(3)
        if x_sum < 0 and y_sum < 0:
            cv2.moveWindow(name, step_x, step_y)
            #return 4
            print(4)
        cv2.imshow(name, self.image)
        return 0

    def get_web_cam(self):
        self.cap = cv2.VideoCapture(0)

    def detect_ratate_head(self, leftPoint, rightPoint):
        dist_between_eyes = (math.sqrt(
            (leftPoint[0] - rightPoint[0]) * (leftPoint[0] - rightPoint[0]) + (leftPoint[1] - rightPoint[1]) * (
                    leftPoint[1] - rightPoint[1])))
        cos_angle = ((leftPoint[0] - rightPoint[0]) / dist_between_eyes)
        if (1 - cos_angle) < 1e-7:
            # normal pose
            sign = 0
        else:
            sign = ((leftPoint[1] - rightPoint[1]) / (abs(leftPoint[1] - rightPoint[1])))

        self.rotate_head = int(math.acos(cos_angle)/math.pi * 180)
        return cos_angle, sign

    def go_browser(self):
        url = 'https://nn.kassir.ru/shou/kontsertnyiy-zal-yupiter/uralskie-pelmeni-luchshee_2020-03-18'
        # Windows
        chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
        webbrowser.get(chrome_path).open(url)

    def get_h_eyes(self):
        return self.h_eyes

    #emotion score
    def emotion_score(self):
        return self.emotion_probs[0] * 2 + self.emotion_probs[2] - self.emotion_probs[1]
        pass


    def add_defects(self, str):
        self.defects.append(str)

    def go_response(self):
        url = 'http://10.70.12.87/update2.php'
        data = {"uid": 1,
                "emoji" : self.how_you_today(),
                "g1": self.hertz,
                "g2": self.h,
                "g3": self.h_eyes,
                "g4": self.rotate_head,
                "g5": int(self.emotion_score()),
                "g6": self.rating}
        print(data)
        response = requests.post(url, data=data)
        print(response.text)
        pass

    def sholders(self, roi_line, t):
        for i, r in enumerate(roi_line):
            if r[0] != 0:
                try:
                    if abs(int(roi_line[i + 1][0] - int(r[0]))) > 40:
                        return i
                except:
                    #print("Fail", i, t)
                    return i
                    pass
        return 0

    def emotion(self, img):
        frame = imutils.resize(img, width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)

        if len(faces) > 0:
            faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[00]
            (x, y, w, h) = faces
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = self.emotion_classifier.predict(roi)[0]
        else:
            return
            pass

        for (i, (emotion, prob)) in enumerate(zip(self.Emotion, preds)):
            if emotion is "happy":
                self.emotion_probs[0] += prob
                self.emotion_prob[0] = prob
            if emotion is "sad":
                self.emotion_probs[1] += prob
                self.emotion_prob[1] += prob
            if emotion is "neutral":
                self.emotion_probs[2] += prob
                self.emotion_prob[2] += prob

        pass

    def score_h(self, h1, h2):
        defense = abs(h1 - h2)
        return defense

    def how_you_today(self):
        if self.emotion_probs[0] > self.emotion_probs[1]:
            if self.emotion_probs[0] > self.emotion_probs[2]:
                print("YOU ARE HAPPY")
                return "happy"
            else:
                print("YOU ARE NEUTRAL")
                return "neutral"
        else:
            if self.emotion_probs[1] > self.emotion_probs[2]:
                print("YOU ARE SAD")
                return "sad"
            else:
                print("YOU ARE NEUTRAL")
                return "neutral"

    def test(self):
        print("test")

    def run(self):
        print("RUN DETECT DEFECT WORKER PLEASE SITDOWN NORMAL")

        self.get_web_cam()
        ret, img = self.cap.read()
        begin = time.time()

        if ret == False:
            print("CAPTURE IS NOT LOAD")
            return

        while True:
            now_time = time.time()
            if now_time - begin > 40:
                begin = now_time
                try:
                    self.go_response()
                except:
                    pass
                self.hertz = int((self.hertz + self.TOTAL)/2)
                print(self.hertz)
                self.TOTAL = 0

            _, img = self.cap.read()
            self.emotion(img)
            img = imutils.resize(img, width=450)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            rects = self.detector(gray, 0)
            for rect in rects:
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[self.lStart:self.lEnd]
                rightEye = shape[self.rStart:self.rEnd]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)

                ear = (leftEAR + rightEAR) / 2.0

                # leftEyeHull = cv2.convexHull(leftEye)
                # rightEyeHull = cv2.convexHull(rightEye)

                # cv2.drawContours(img, [leftEyeHull], -1, (0, 0, 255), 1)
                # cv2.drawContours(img, [rightEyeHull], -1, (0, 0, 255), 1)

                if ear > self.EYE_AR_THRESH:
                    self.COUNTER += 1
                else:
                    if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                        self.TOTAL += 1

                    self.Counter = 0

                rotate, sign = self.detect_ratate_head(leftEye[0], rightEye[1])
                if abs(rotate) < 0.98:
                    if sign > 0:
                        self.rotate_str = "Bad Rotate Head(Left)"
                    else:
                        self.rotate_str = "Bad Rotate Head(Right)"
                else:
                    self.rotate_str = ""

                cv2.putText(img, "Blink: {}".format(self.TOTAL), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(img, self.rotate_str, (150, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            center_left = (leftEye[0][0] + int((leftEye[3][0] - leftEye[0][0]) / 2))
            center_right = (rightEye[0][0] + int((rightEye[3][0] - rightEye[0][0]) / 2))

            center_x = center_left + int((center_right - center_left) / 2)

            left_x, right_x = center_x - 90, center_x + 90
            hight = leftEye[0][1]
            if hight < rightEye[0][1]:
                hight = rightEye[0][1]

            self.h_eyes = hight

            roi_line = gray[hight + 50:gray.shape[0], left_x:left_x + 1]
            roi_line = roi_line[::-1]

            roi_line2 = gray[hight + 50:gray.shape[0], right_x:right_x + 1]
            roi_line2 = roi_line2[::-1]

            key = self.sholders(roi_line, 1)
            key1 = self.sholders(roi_line2, 2)

            self.h = self.score_h(key, key1)
            if abs(key1 - key) > 5 or key == 0 or key1 == 0:
                if self.iteration > 7:
                    cv2.putText(img, "Bad Pose", (300, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                else:
                    self.iteration += 1
            else:
                self.iteration = 0

            # cv2.line(img, (0, gray.shape[0] - key), (gray.shape[1], gray.shape[0] - key), (0, 0, 255))
            # cv2.line(img, (0, gray.shape[0] - key1), (gray.shape[1], gray.shape[0] - key1), (255, 255, 255))

            # cv2.imshow("GRAY", roi_line)
            self.form_rate()
            cv2.imshow("result", img)
            key = cv2.waitKey(5)

            if key == 27:
                print(self.emotion_probs)
                print(self.rating)
                self.how_you_today()

                self.go_browser()
                break
            if key == 113:
                self.go_response()
                print(leftEye[0], rightEye[3])
                print(self.detect_ratate_head(leftEye[0], rightEye[3]))

        cv2.destroyAllWindows()
        self.cap.release()

    def run2(self):
        print("RUN DETECT DEFECT WORKER PLEASE SITDOWN NORMAL")

        self.get_web_cam()
        ret, img = self.cap.read()

        if ret == False:
            print("CAPTURE IS NOT LOAD")
            return
        # для накопления положения зрачка
        diff_pupil_pos_x = []
        diff_pupil_pos_y = []

        self.image = cv2.imread("cat.jpeg")

        while True:
            _, img = self.cap.read()
            img = imutils.resize(img, width=450)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            rects = self.detector(gray, 0)
            for rect in rects:
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[self.lStart:self.lEnd]
                rightEye = shape[self.rStart:self.rEnd]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)

                ear = (leftEAR + rightEAR) / 2.0

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)

                cv2.drawContours(img, [leftEyeHull], -1, (0, 0, 255), 1)
                cv2.drawContours(img, [rightEyeHull], -1, (0, 0, 255), 1)

                if ear > self.EYE_AR_THRESH:
                    self.COUNTER += 1
                else:
                    if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                        self.TOTAL += 1

                    self.Counter = 0

                # копим положение зрачка и центра глаза
                diff_pupil_pos_x_current, diff_pupil_pos_y_current = self.get_diff_position_eye_pupil(gray, leftEye)
                if abs(diff_pupil_pos_x_current) < 10e4: # выбросы убираем
                    diff_pupil_pos_x.append(diff_pupil_pos_x_current)
                    diff_pupil_pos_y.append(diff_pupil_pos_y_current)

                if len(diff_pupil_pos_x) == 6:
                    number_of_quadrant = self.get_number_of_quadrant(diff_pupil_pos_x, diff_pupil_pos_y)
                    print("number_of_quadrant = ", number_of_quadrant)
                    diff_pupil_pos_x = []
                    diff_pupil_pos_y = []

                # анализ поворота головы
                rotate, sign = self.detect_ratate_head(leftEye[0], rightEye[1])
                if rotate < 0.98:
                    if sign > 0:
                        self.rotate_str = "No normal(left)"
                    else:
                        self.rotate_str = "No normal(right)"
                else:
                    self.rotate_str = "Normal"

                cv2.putText(img, "Blink: {}".format(self.TOTAL), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(img, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(img, self.rotate_str, (150, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("result", img)
            key = cv2.waitKey(5)

            if key == 27:
                break
            if key == 113:
                print(leftEye[0], rightEye[3])
                print(self.detect_ratate_head(leftEye[0], rightEye[3]))

        cv2.destroyAllWindows()
        self.cap.release()


def run():
    work = Worker()
    work.run()

def run2():
    work = Worker()
    work.run2()

if __name__ == "__main__":
    work = Worker()
    work.run2()
