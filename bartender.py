# INTINT WITH THE BOYS
# HUGO ASZTELY, SIMON OLOFSSON & JOEL TINGWALL
# Code outline for a hardcoded furhat bartender with dialogues based on keywords

# change to your directory in terminal where the python used is, and pip install all necessary modules and references
# NECESSARY IMPORTS
from time import sleep
from furhat_remote_api import FurhatRemoteAPI
from numpy.random import randint
import speech_recognition as sr
import random

FURHAT_IP = "127.0.1.1"

furhat = FurhatRemoteAPI(FURHAT_IP)
furhat.set_led(red=100, green=50, blue=50)

# Initialize the speech recognition recognizer
recognizer = sr.Recognizer()

FACES = {
    'Amany': 'Nazar'
}

VOICES_EN = {
    'Amany': 'CoraNeural'
}

VOICES_NATIVE = {
    'Amany': 'AmanyNeural'
}

def idle_animation():
    furhat.gesture(name="GazeAway")
    gesture = {
        "frames": [
            {
                "time": [0.33],
                "persist": True,
                "params": {
                    "NECK_PAN": randint(-4, 4),
                    "NECK_TILT": randint(-4, 4),
                    "NECK_ROLL": randint(-4, 4),
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

# DO NOT CHANGE
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

# DIALOGUE KEYWORDS THATAMANI CAN RESPOND TO
KEYWORD_CAN_I_GET_A = "can i get a"
KEYWORD_WAIT = "wait" 
KEYWORD_LEAVING = "bye"

# DIALOGUE CONSTANTS THAT AMANY SAYS UPON INTRO AND ENDING
KEYWORD_GREET = "Hello client number 8. What can I get you?" # AMANI
KEYWORD_BYE= "Okay, please come back, or don't. I literally cannot care less." # AMANI

def analyze_keywords(user_response):
    keyword_responses = {
        KEYWORD_CAN_I_GET_A: "Certainly, coming right up in ... calculating... six hours. Please transaction 0.00002 bitcoin.",
        KEYWORD_WAIT: "Yes I am aware that service is a little slow. I dont have any arms to mix drinks with.",
    }

    # RANDOM RESPONSES IF SENTENCE USER SAYS DOES NOT MATCH KEYWORD
    default_responses = [
        "I don't speak drunk. Try to improve your voice mechanics.",
        "What did you say?"
    ]

    for keyword, response in keyword_responses.items():
        if keyword in user_response.lower():
            if keyword == KEYWORD_CAN_I_GET_A:
                response_parts = response.split("Certainly,")
                if len(response_parts) == 2:
                    # Perform the "Wink" gesture after saying "Certainly,"
                    bsay("Certainly,")
                    furhat.gesture(name="Wink")
                    sleep(0.3)  # Adjust the sleep time as needed
                    return response_parts[1].strip()
            elif keyword == KEYWORD_WAIT:
                response_parts = response.split("little slow")
                if len(response_parts) == 2:
                    # Perform the "Thoughtful" gesture after saying "little slow"
                    bsay("Yes I am aware that service is a little slow.")
                    sleep(0.5)  # Adjust the sleep time as needed
                    furhat.gesture(name="Thoughtful")
                    sleep(0.5)  # Adjust the sleep time as needed
                    return response_parts[1].strip()

    # If no keyword matched, select a random default response from default_responses
    random_default_response = random.choice(default_responses)
    
    # Perform the "Roll" gesture
    furhat.gesture(name="Roll")
    
    return random_default_response

def interact_with_user():
    set_persona('Amany')
    bsay(KEYWORD_GREET)

    while True:
        user_response = listen_to_user()

        exit_phrases = [KEYWORD_LEAVING]

        
        # Check if any of the exit phrases are present in the user's response
        if any(phrase in user_response.lower() for phrase in exit_phrases):
            bsay(KEYWORD_BYE)
            furhat.gesture(name="Roll")
            break

        furhat.set_voice(name=VOICES_NATIVE['Amany'])

        response = analyze_keywords(user_response)

        bsay(response)

def listen_to_user():
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source)
            user_response = recognizer.recognize_google(audio)  # Recognize user's speech
            print(f"User said: {user_response}")
            return user_response
        except sr.UnknownValueError:
            return ""  # Return an empty string if speech couldn't be recognized

if __name__ == '__main__':
    interact_with_user()
    idle_animation()

