# Import necessary modules
import cv2
import numpy as np
import random
import speech_recognition as sr
from furhat_remote_api import FurhatRemoteAPI
from tensorflow import keras
from time import sleep

# Load the pre-trained emotion recognition model
emotion_model = keras.models.load_model('emotion_recognition_model.h5')

# Define a list of emotions for classification
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Define Furhat parameters
FURHAT_IP = "127.0.1.1"
furhat = FurhatRemoteAPI(FURHAT_IP)
furhat.set_led(red=100, green=50, blue=50)

# Initialize the speech recognition recognizer
recognizer = sr.Recognizer()

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

# Define a mapping of emotions to responses
emotion_responses = {
    "angry": "I see you're angry. What can I do to help?",
    "happy": "Glad you are feeling happy!",
    "sad": "I'm here to listen. What's bothering you?",
    "surprise": "You look surprised! What's going on?",
    "disgust": "I sense some disgust. Can you tell me what's bothering you?",
    "fear": "Don't be afraid. I'm here to assist you. What can I help you with?",
    "neutral": "I'm here to assist you. How can I help?"
}

# DIALOGUE KEYWORDS THAT AMANI CAN RESPOND TO
KEYWORD_CAN_I_GET_A = "can I get a"
KEYWORD_WAIT = "wait"
KEYWORD_LEAVING = "bye"

# DIALOGUE CONSTANTS THAT AMANY SAYS UPON INTRO AND ENDING
KEYWORD_GREET = "Hello! What can I get you?"  # AMANI
KEYWORD_BYE = "Okay, please come back, or don't. I literally cannot care less."  # AMANI

def analyze_keywords(user_response):
    user_response = user_response.lower()

    keyword_responses = {
        KEYWORD_CAN_I_GET_A: "Certainly, coming right up in six hours. Please transaction 0.00002 bitcoin.",
        KEYWORD_WAIT: "Yes, I am aware that service is a little slow. I don't have any arms to mix drinks with.",
    }

    default_responses = [
        "I don't speak drunk. Try to improve your voice mechanics.",
        "What did you say?"
    ]

    for keyword, response in keyword_responses.items():
        if keyword.lower() in user_response:
            if keyword == KEYWORD_CAN_I_GET_A:
                response_parts = response.split("Certainly,")
                if len(response_parts) == 2:
                    bsay("Certainly,")
                    furhat.gesture(name="Wink")
                    sleep(0.3)
                    return response_parts[1].strip()
            elif keyword == KEYWORD_WAIT:
                response_parts = response.split("little slow")
                if len(response_parts) == 2:
                    bsay("Yes, I am aware that service is a little slow.")
                    sleep(0.5)
                    furhat.gesture(name="Thoughtful")
                    sleep(0.5)
                    return response_parts[1].strip()

    random_default_response = random.choice(default_responses)
    furhat.gesture(name="Roll")
    return random_default_response

# Function to recognize emotions in real-time from video feed
def recognize_emotion_from_video():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        input_frame = cv2.resize(frame, (48, 48))
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        input_frame = np.expand_dims(input_frame, axis=0)

        emotion_predictions = emotion_model.predict(input_frame)
        emotion_label = emotion_labels[emotion_predictions[0].argmax()]
        print(emotion_predictions)
        return emotion_label

    cap.release()
    cv2.destroyAllWindows()

# Function to analyze emotions and provide responses
def analyze_emotion(emotion):
    return emotion_responses.get(emotion.lower(), "I'm here to assist you. How can I help?")

# Function to interact with the user
def interact_with_user():
    set_persona('Amany')
    bsay(KEYWORD_GREET)

    while True:
        user_response = listen_to_user()

        # Capture the emotion only once
        recognized_emotion = recognize_emotion_from_video()

        exit_phrases = [KEYWORD_LEAVING]

        if any(phrase in user_response.lower() for phrase in exit_phrases):
            bsay(KEYWORD_BYE)
            furhat.gesture(name="Roll")
            break

        furhat.set_voice(name=VOICES_NATIVE['Amany'])

        response = analyze_keywords(user_response)
        print(f"Amani's Keyword Response: {response}")  # Print Amani's response to the keyword
        bsay(response)
        sleep(1)

        emotion_response = analyze_emotion(recognized_emotion)
        print(f"Recognized Emotion: {recognized_emotion}")  # Print the recognized emotion
        print(f"Amani's Response: {emotion_response}")  # Print Amani's response to the emotion
        bsay(emotion_response)  # Say the recognized emotion
# Function to listen to user's speech
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
