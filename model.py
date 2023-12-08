import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

data_directory = "DiffusionFER/DiffusionEmotion_S_cropped"

# Define a function to load and preprocess the images
def load_images_from_folder(folder):
    images = []
    labels = []
    for emotion_folder in os.listdir(folder):
        emotion_path = os.path.join(folder, emotion_folder)
        emotion_label = emotion_folder  # The emotion label is the folder name
        for filename in os.listdir(emotion_path):
            img = cv2.imread(os.path.join(emotion_path, filename))
            if img is not None:
                images.append(img)
                labels.append(emotion_label)
    return images, labels

images, labels = load_images_from_folder(data_directory)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Resize and normalize the images
image_size = (48, 48)
X_train = [cv2.resize(image, image_size) for image in X_train]
X_test = [cv2.resize(image, image_size) for image in X_test]
print(image_size)
X_train = np.array(X_train) #/ 255.0, got better performance without this part
X_test = np.array(X_test) #/ 255.0

from sklearn.preprocessing import LabelEncoder

# Encode emotion labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


from tensorflow.keras import layers, models

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(7, activation='softmax')  # 7 classes for the 7 emotions
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_encoded, epochs=15, validation_data=(X_test, y_test_encoded))

test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
print(f'Test accuracy: {test_acc}')

model.save('emotion_recognition_model.h5')
