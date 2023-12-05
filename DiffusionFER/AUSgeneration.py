#############################################################################################################
# THIS SCRIPT IS ONLY USED WHEN GENERATING THE AUS DATAFRAME aus_data.csv. IT WILL TAKE A LONG TIME TO RUN SINCE IT ITERATIES OVER A 1000 PICTURES
# THE aus.csv FILE SHOULD ALREADY EXIST IN THE FOLDER. IF YOU WISH TO RUN THIS SCRIPT PLEASE UNCOMMENT THE TINY LOOP OF: if count == limit_per_folder: break AROUND LINE 63
##############################################################################################################

import pandas as pd
import os
from feat import Detector

#Current directory
current_path = os.getcwd()

#Output path for saving the action units/emotions/filenames dataframe
csv_output_path = os.path.join(current_path, 'aus_data.csv')

#Loading dataset sheet csv file
csv_file_path = os.path.join(current_path, 'dataset_sheet.csv')
dataset_df = pd.read_csv(csv_file_path)

#List of emotions and also name of folders
image_folders = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'angry']

#Detector initialization
detector = Detector(device="cpu")

#Dataframe for AUs
aus_df = pd.DataFrame()
aus_df.insert(0, "emotion", "")
aus_df.insert(1, "file", 0)

# List to store image paths and labels
image_paths = []
labels = []

# Limit of images to process per folder, this is needed for testing since it would take forever to process all images for each run
limit_per_folder = 1

#Initial loop for the seven emotion folder
for folder in image_folders:
    #path to current folder
    folder_path = os.path.join(current_path, 'DiffusionEmotion_S_cropped', folder)

    # Counter for processed images. for testing purposes
    count = 0

    #Second loop for iterating through the images in the folder and giving AU values to each image in order to build the dataframe.
    for f in os.listdir(folder_path):
        if f.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            #Complete image path
            image_path = os.path.join(folder_path, f)

            #Verify that the file exists
            if os.path.exists(image_path):
                image_paths.append(image_path)
                labels.append(folder)

                #Get the AU values for the image
                aus_values = detector.detect_image(image_path).aus

                #Extracting valence and arousal from dataset_df
                image_info = dataset_df[dataset_df['subDirectory_filePath'].str.contains(f)]
                valence = image_info['valence'].values[0]
                arousal = image_info['arousal'].values[0]

                #Create a dataframe with the AU values, valence, arousal, emotion and file name
                au = aus_values.loc[[0]]
                au['valence'] = valence
                au['arousal'] = arousal
                au['emotion'] = folder
                au['file'] = os.path.splitext(f)[0]

                #Concatenate au dataframe to the main dataframe
                aus_df = pd.concat([aus_df, au], ignore_index=True)

                #Increment the counter
                count += 1

                #Break out of the loop if the limit is reached ######### UNCOMMENT THIS IF YOU WANT TO RUN THE SCRIPT
                # if count == limit_per_folder:
                #     break
            else:
                print(f"Warning: Image not found - {image_path}")
print(aus_df)

#Save the dataframe to a csv file
aus_df.to_csv(csv_output_path, index=False)