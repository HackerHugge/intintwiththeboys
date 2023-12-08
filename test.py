import cv2
import numpy as np
from tensorflow import keras

# Load the pre-trained emotion recognition model
emotion_model = keras.models.load_model('emotion_recognition_model.h5')

# Define an emotion mapping (optional)
emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Load an input image (replace 'your_image.jpg' with the actual image file)
input_image = cv2.imread('your_image.jpg')

# Convert the image to grayscale
input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Resize the image to (48, 48) as expected by the model
input_gray = cv2.resize(input_gray, (48, 48))

# Normalize the pixel values to be in the range [0, 1]
input_gray = input_gray / 255.0

# Expand dimensions to match model input shape (batch size of 1)
input_gray = np.expand_dims(input_gray, axis=0)
input_gray = np.expand_dims(input_gray, axis=-1)

# Perform emotion prediction
emotion_predictions = emotion_model.predict(input_gray)

# Get the predicted emotion label
predicted_emotion_label = np.argmax(emotion_predictions)

# Map the label to the corresponding emotion (optional)
predicted_emotion = emotion_mapping.get(predicted_emotion_label, "Unknown")

# Print the predicted emotion
print(f"Predicted Emotion: {predicted_emotion}")
