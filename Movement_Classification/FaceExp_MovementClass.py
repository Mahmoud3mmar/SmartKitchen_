import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

import mediapipe as mp  # Import mediapipe
import cv2  # Import opencv
from dollarpy import Recognizer, Template, Point
# import csv
import os
import numpy as np
import matplotlib.pyplot as plt
# from SocketConnection import connect
mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
mp_holistic = mp.solutions.holistic  # Mediapipe Solutions
import pickle

import cv2
import mediapipe as mp
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from dollarpy import Point
import matplotlib.pyplot as plt
#
# class EmotionDetector:
#     def __init__(self):
#         self.face_classifier = cv2.CascadeClassifier(
#             r'D:\HCI\SmartKitchen\Emotion_Detection\haarcascade_frontalface_default.xml')
#         self.classifier = load_model(r'D:\HCI\SmartKitchen\Emotion_Detection\model.h5')
#         self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
#         self.mp_drawing = mp.solutions.drawing_utils
#         self.mp_holistic = mp.solutions.holistic
#         self.points = []
#
#     def detect_emotions(self):
#         cap = cv2.VideoCapture(0)
#         with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 labels = []
#                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 faces = self.face_classifier.detectMultiScale(gray)
#                 if not ret:
#                     continue
#
#                 image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 image.flags.writeable = False
#                 results = holistic.process(image)
#                 image.flags.writeable = True
#                 image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#                 self.display_landmarks(frame, results)
#
#                 for (x, y, w, h) in faces:
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
#                     roi_gray = gray[y:y + h, x:x + w]
#                     roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
#
#                     if np.sum([roi_gray]) != 0:
#                         roi = roi_gray.astype('float') / 255.0
#                         roi = img_to_array(roi)
#                         roi = np.expand_dims(roi, axis=0)
#
#                         prediction = self.classifier.predict(roi)[0]
#                         label = self.emotion_labels[prediction.argmax()]
#                         label_position = (x, y)
#                         cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                     else:
#                         cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#                 cv2.imshow('Emotion Detector', frame)
#                 if cv2.waitKey(1) & 0xFF == ord('x'):
#                     break
#
#         cap.release()
#         cv2.destroyAllWindows()
#         points = self.get_landmarks(results)
#         print("All Landmarks")
#         self.plot_landmarks(points)
#
#         return points
#
#     def display_landmarks(self, frame, results):
#         if results.pose_landmarks:
#             mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                                       mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
#                                       mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
#                                       )
#
#             # 3. Left Hand
#             mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                                       mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
#                                       mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
#                                       )
#
#             # 4. Pose Detections
#             mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
#                                       mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
#                                       mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
#                                       )
#
#     def get_landmarks(self, results):
#         landmark_points = []
#         if results.pose_landmarks:
#             for landmark in results.pose_landmarks.landmark:
#                 landmark_points.append((landmark.x, landmark.y))
#         return landmark_points
#
#     def plot_landmarks(self, points):
#         xs, ys = zip(*points)
#         plt.plot(xs, ys, 'o')
#         plt.plot(xs, ys, '-')
#         plt.gca().invert_yaxis()
#         # plt.show()
# # # if __name__ == "__main__":
# # #     face_cascade_path = r'C:\Users\mahmo\Downloads\Compressed\Emotion_Detection\haarcascade_frontalface_default.xml'
# # #     model_path = r'C:\Users\mahmo\Downloads\Compressed\Emotion_Detection\model.h5'
# # # from Detectionclass import EmotionDetector
# # #
# # emotion_detector = EmotionDetector()
# # emotion_detector.detect_emotions()
# #
# if __name__ == "__main__":
#     # Load the templates array from the file
#     with open('templates.pkl', 'rb') as file:
#         loaded_templates = pickle.load(file)
#     # Create a recognizer and use it to classify gestures
#     recognizer = Recognizer(loaded_templates)
#     emotion_detector = EmotionDetector()
#
#     points_say_hello = emotion_detector.detect_emotions()
#     print(points_say_hello)
#     result = recognizer.recognize(points_say_hello)
#
#     print(result[0])


import mediapipe as mp
import cv2
from dollarpy import Point
import matplotlib.pyplot as plt

import cv2
import mediapipe as mp
from dollarpy import Point
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
class HandGestureCapture:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.left_shoulder = []
        self.right_shoulder = []
        self.left_elbow = []
        self.right_elbow = []
        self.left_wrist = []
        self.right_wrist = []
        self.left_pinky = []
        self.right_pinky = []
        self.left_index = []
        self.right_index = []
        self.left_hip = []
        self.right_hip = []

        self.m_left_shoulder = []
        self.m_right_shoulder = []
        self.m_left_elbow = []
        self.m_right_elbow = []
        self.m_left_wrist = []
        self.m_right_wrist = []
        self.m_left_pinky = []
        self.m_right_pinky = []
        self.m_left_index = []
        self.m_right_index = []
        self.m_left_hip = []
        self.m_right_hip = []

        self.face_classifier = cv2.CascadeClassifier(
            r'C:\Users\mahmo\Downloads\Compressed\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')
        self.classifier = load_model(r'C:\Users\mahmo\Downloads\Compressed\Emotion_Detection_CNN-main\model.h5')
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def capture_video_points(self):
        cap = cv2.VideoCapture(0)
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                labels = []
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces using the cascade classifier
                faces = self.face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if ret:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = holistic.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # Draw landmarks
                    self._draw_landmarks(image, results)

                    # Extract and store points
                    self._extract_and_store_points(results.pose_landmarks)

                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        roi_gray = gray[y:y + h, x:x + w]
                        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                        if np.sum([roi_gray]) != 0:
                            roi = roi_gray.astype('float') / 255.0
                            roi = img_to_array(roi)
                            roi = np.expand_dims(roi, axis=0)

                            prediction = self.classifier.predict(roi)[0]
                            label = self.emotion_labels[prediction.argmax()]
                            label_position = (x, y)
                            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.imshow("Gesture", frame)

                    if cv2.waitKey(10) & 0xFF == ord('x'):
                        break

        cap.release()
        cv2.destroyAllWindows()
        return self._get_all_points()

    def _draw_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                       )
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                       )
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                       )

    def _extract_and_store_points(self, pose_landmarks):
        if pose_landmarks:
            newlist = [pose_landmarks.landmark[i] for i in [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24]]

            # add points of wrist, elbow, and shoulder
            self.left_shoulder.append(Point(newlist[0].x, newlist[0].y, 1))
            self.right_shoulder.append(Point(newlist[1].x, newlist[1].y, 2))
            self.left_elbow.append(Point(newlist[2].x, newlist[2].y, 3))
            self.right_elbow.append(Point(newlist[3].x, newlist[3].y, 4))
            self.left_wrist.append(Point(newlist[4].x, newlist[4].y, 5))
            self.right_wrist.append(Point(newlist[5].x, newlist[5].y, 6))
            self.left_pinky.append(Point(newlist[6].x, newlist[6].y, 7))
            self.right_pinky.append(Point(newlist[7].x, newlist[7].y, 8))
            self.left_index.append(Point(newlist[8].x, newlist[8].y, 9))
            self.right_index.append(Point(newlist[9].x, newlist[9].y, 10))
            self.left_hip.append(Point(newlist[10].x, newlist[10].y, 11))
            self.right_hip.append(Point(newlist[11].x, newlist[11].y, 12))

            self.m_left_shoulder.append((newlist[0].x, newlist[0].y))
            self.m_right_shoulder.append((newlist[1].x, newlist[1].y))
            self.m_left_elbow.append((newlist[2].x, newlist[2].y))
            self.m_right_elbow.append((newlist[3].x, newlist[3].y))
            self.m_left_wrist.append((newlist[4].x, newlist[4].y))
            self.m_right_wrist.append((newlist[5].x, newlist[5].y))
            self.m_left_pinky.append((newlist[6].x, newlist[6].y))
            self.m_right_pinky.append((newlist[7].x, newlist[7].y))
            self.m_left_index.append((newlist[8].x, newlist[8].y))
            self.m_right_index.append((newlist[9].x, newlist[9].y))
            self.m_left_hip.append((newlist[10].x, newlist[10].y))
            self.m_right_hip.append((newlist[11].x, newlist[11].y))

    def _get_all_points(self):
        return (
                self.left_shoulder + self.right_shoulder +
                self.left_elbow + self.right_elbow +
                self.left_wrist + self.right_wrist +
                self.left_pinky + self.right_pinky +
                self.left_index + self.right_index +
                self.left_hip + self.right_hip
        )

    def plot_points(self):
        m_points = [(point.x, point.y) for point in self.left_shoulder]
        xs, ys = zip(*m_points)
        plt.plot(xs, ys, 'o')
        plt.plot(xs, ys, '-')
        plt.show()

# if __name__ == "__main__":
#         # Load the templates array from the file
#         with open('templates.pkl', 'rb') as file:
#             loaded_templates = pickle.load(file)
#         # Create a recognizer and use it to classify gestures
#         recognizer = Recognizer(loaded_templates)
#         hand_gesture_capture = HandGestureCapture()
#         points_say_hello = hand_gesture_capture.capture_video_points()
#         result = recognizer.recognize(points_say_hello)
#
#         print(result[0])
    # hand_gesture_capture = HandGestureCapture()
    # captured_points = hand_gesture_capture.capture_video_points()
    # print("Captured Points:", captured_points)
    # hand_gesture_capture.plot_points()
