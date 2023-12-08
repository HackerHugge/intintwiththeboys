import cv2
import numpy as np
from feat import Detector
import os
import joblib
import pandas as pd
import random
import speech_recognition as sr
from furhat_remote_api import FurhatRemoteAPI
from time import sleep
import time
import threading
import warnings

# Suppress the specific UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.functional")

# Load your trained CNN model
model_filename = "valence_regression_model.joblib"
model_path = os.path.join(os.getcwd(), model_filename)
model = joblib.load(model_path)

# Create a PyFeat detector
detector = Detector()

# Define Furhat parameters
FURHAT_IP = "127.0.1.1"
furhat = FurhatRemoteAPI(FURHAT_IP)
furhat.set_led(red=100, green=50, blue=50)

# Initialize the speech recognition recognizer
recognizer = sr.Recognizer()

def initialize_video_capture():
    return cv2.VideoCapture(0)

# Define Furhat attributes
FACES = {
    'Amany': 'Nazar'
}

VOICES_EN = {
    'Amany': 'CoraNeural'
}

VOICES_NATIVE = {
    'Amany': 'AmanyNeural'
}

# Define Furhat gestures
def idle_animation():
    furhat.gesture(name="GazeAway")
    gesture = {
        "frames": [
            {
                "time": [0.33],
                "persist": True,
                "params": {
                    "NECK_PAN": random.randint(-4, 4),
                    "NECK_TILT": random.randint(-4, 4),
                    "NECK_ROLL": random.randint(-4, 4),
                }
            }
        ],
        "class": "furhatos.gestures.Gesture"
    }
    furhat.gesture(body=gesture, blocking=True)

def LOOK_BACK(speed):
    return {
        "frames": [
            {
                "time": [
                    0.33 / speed
                ],
                "persist": True,
                "params": {
                    'LOOK_DOWN': 0,
                    'LOOK_UP': 0,
                    'NECK_TILT': 0
                }
            }, {
                "time": [
                    1 / speed
                ],
                "params": {
                    "NECK_PAN": 0,
                    'LOOK_DOWN': 0,
                    'LOOK_UP': 0,
                    'NECK_TILT': 0
                }
            }
        ],
        "class": "furhatos.gestures.Gesture"
    }

def LOOK_DOWN(speed=1):
    return {
        "frames": [
            {
                "time": [
                    0.33 / speed
                ],
                "persist": True,
                "params": {
                    # 'LOOK_DOWN' : 1.0
                }
            }, {
                "time": [
                    1 / speed
                ],
                "persist": True,
                "params": {
                    "NECK_TILT": 20
                }
            }
        ],
        "class": "furhatos.gestures.Gesture"
    }

def set_persona(persona):
    furhat.gesture(name="CloseEyes")
    furhat.gesture(body=LOOK_DOWN(speed=1), blocking=True)
    sleep(0.3)
    furhat.set_face(character=FACES[persona], mask="Adult")
    furhat.set_voice(name=VOICES_EN[persona])
    sleep(2)
    furhat.gesture(body=LOOK_BACK(speed=1), blocking=True)

# Say with blocking (blocking say, bsay for short)
def bsay(line):
    furhat.say(text=line, blocking=True)

def interact_with_user():
    set_persona('Amany')
    cap = initialize_video_capture()
    while True:

        user_response = listen_to_user()

        recognized_valence = recognize_valence_from_video(4, cap)
        
        # Convert the list of arrays to a numpy array
        valence_array = np.array(recognized_valence)
        print("Valence array:", valence_array)
        # Calculate the mean along the specified axis (0 for column-wise, 1 for row-wise)
        mean_valence = np.mean(valence_array, axis=0)

        print("Mean valence:", mean_valence)

        sleep(3)
        if mean_valence > 0.2:
            bsay("I see you are happy!")
        elif mean_valence < -0.1:
            bsay("I see you are sad!")
        else:
            bsay("I see you are neutral!")
        


def recognize_valence_from_video(duration_seconds, cap):
    start_time = time.time()
    valence_list = []
    while True:
        print("Watching...", end="\r")
        # Capture frame-by-frame
        ret, frame = cap.read()

        faces = detector.detect_faces(frame)

        if not faces:
            print("No faces found.")
            # You can add some code here to handle this case, like displaying a message
            # or taking specific actions
            # For example, you might continue to the next iteration of the loop:
            continue

        landmarks = detector.detect_landmarks(frame, faces)

        # Preprocess the features to match the input format of your model
        # (resize, normalize, etc.)
        au_values = detector.detect_aus(frame, landmarks)

        au_values_flat = au_values[0][0]

        # Make predictions using your trained model
        valence_prediction = model.predict([au_values_flat])
        # Display the resulting frame with valence information
        # (e.g., draw the valence prediction on the frame)

        # 

        valence_list.append(valence_prediction)

        elapsed_time = time.time() - start_time
        if elapsed_time >= duration_seconds:
            break

    print("Watching... Done!")
    # Release the video capture object
    # cap.release()

    return valence_list


def listen_to_user():
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source)
            user_response = recognizer.recognize_google(audio)
            print(f"User said: {user_response}")
            return user_response
        except sr.UnknownValueError:
            return ""

if __name__ == '__main__':
    try:
        interact_with_user()
    finally:
        idle_animation()