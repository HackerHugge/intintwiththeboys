import cv2
import numpy as np
from feat import Detector
import os
import joblib
import random
import speech_recognition as sr
from furhat_remote_api import FurhatRemoteAPI
from time import sleep
import time
import warnings

# Suppress the specific UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.functional")

# Load trained model via joblib
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

#Function to initialize video capture
def initialize_video_capture():
    return cv2.VideoCapture(0)

# Define Furhat attributes
FACES = {
    'Amany': 'Isabel'
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

#Initialize Furhat persona
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
      

#Function to recognize valence from input images (frames)
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
            continue

        # Detect landmarks and AUs for each face
        landmarks = detector.detect_landmarks(frame, faces)
        au_values = detector.detect_aus(frame, landmarks)
        au_values_flat = au_values[0][0]

        # Make predictions using our trained model
        valence_prediction = model.predict([au_values_flat])
       
        # Append the prediction to the list, So that we later can calculate the mean valence and get a more precise result
        valence_list.append(valence_prediction)

        elapsed_time = time.time() - start_time
        if elapsed_time >= duration_seconds:
            break

    print("Watching... Done!")

    return valence_list

#Function to listen to user that outputs the user response in a string
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
        

#Functions to respond to user based on the keyword
        
def happy_response(keyword):
    print("Keyword:", keyword)
    if keyword == "greet":
        response = "Hey party dog! What's your poison?"
    elif keyword == "order":
        response = "That's a fine choice! Coming right up!"
    elif keyword == "empty":
        response = "This is a virtual bar, my friend!"
    elif keyword == "drink":
        response = "Bro, you know I don't have any arms! I can't make you a drink. You suddenly seem entertained!"
    elif keyword == "thank":
        response = "No worries, mate!"
    elif keyword == "pay":
        response = "I'm programmed to like toxic sad men! You can pay by giving me your number, babe!"
        furhat.gesture(name="Wink")
    elif keyword == "bye":
        response = "See you after my shift, don't tell my boyfriend ChatGPT!"

    return response

def sad_response(keyword):
    print("Keyword:", keyword)
    if keyword == "greet":
        response = "Hey party pooper! What's your numbing drink of choice?"
    elif keyword == "order":
        response = "Sure, I'll give you a free shot on the house. Hopefully it cheers you up, after rain comes moonshine!"
    elif keyword == "empty":
        response = "This is a virtual bar. You should try visiting a real one, you look like you need it."
    elif keyword == "drink":
        response = "I'm sorry, I don't have any arms. Go to a real bar instead."
    elif keyword == "thank":
        response = "My pleasure, I wish you well."
    elif keyword == "pay":
        response = "You can pay with a smile"
    elif keyword == "bye":
        response = "Alright man, I'll see you around. Don't do anything stupid, you're never alone"
    return response

def neutral_response(keyword):
    print("Keyword:", keyword)
    if keyword == "greet":
        response = "Hey man! What can I get you?"
    elif keyword == "order":
        response = "Sure thing, man!"
    elif keyword == "empty":
        response = "This is a virtual bar!"
    elif keyword == "drink":
        response = "I do not have any arms, as you can see. Thus I can only give you a hypothetical drink"
    elif keyword == "thank":
        response = "You're welcome"
    elif keyword == "pay":
        response = "That will be 1 bitcoin, please"
    elif keyword == "bye":
        response = "Goodbye!"

    return response
        
#Analyzing keywords from user response, outputs keyword as a string which is used in the response functions
def analyze_keywords(user_response):
    user_response = user_response.lower()

    # Define keywords
    keywords = {
        'greet': ['hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening', "what's up"],
        'order': ['shot', 'beer', 'order', 'shots', 'wine','beers'],
        'pay': ['pay', 'bill', 'check', 'pay', 'paying'],
        'bye': ['bye', 'goodbye', 'see you', 'later', 'see you later', 'bye bye', 'good night'],
        'thank': ['thank', 'thanks', 'thank you', 'thanks a lot', 'thanks a bunch', 'thank you very much'],
        'empty': ['empty', 'people', 'lonely'],
        'drink': ["where's", "where is", "when", 'drink']
    }

    # Iterate over the keywords and check if any keyword is present in the user response
    for keyword, keyword_list in keywords.items():
        for word in keyword_list:
            if word in user_response:
                return keyword

    return ""

#Function for overall pipeline
def interact_with_user():
    set_persona('Amany')
    cap = initialize_video_capture()
    while True:

        user_response = listen_to_user()

        recognized_valence = recognize_valence_from_video(2, cap)
        
        # Convert the list of arrays to a numpy array
        valence_array = np.array(recognized_valence)
        # print("Valence array:", valence_array)

        # Calculate the mean valence
        mean_valence = np.mean(valence_array, axis=0)

        # print("Mean valence:", mean_valence)

        sleep(1)

        keyword = analyze_keywords(user_response)

        if keyword == "":
            bsay("Mate you're drunk, stop slurring your words!")
        else:
            if mean_valence > 0.2:
                bsay(happy_response(keyword))
                furhat.gesture(name="BigSmile")
            elif mean_valence < -0.1:
                bsay(sad_response(keyword))
                furhat.gesture(name="Thoughtful")
            else:
                bsay(neutral_response(keyword))
                
        
        if keyword == "bye":
            break  # exit the loop when the keyword is "bye"

if __name__ == '__main__':
    try:
        interact_with_user()
    finally:
        idle_animation()