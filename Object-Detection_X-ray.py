#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:45:27 2025

@author: Mert Basaran
"""
# Object  Detection
# Localization of common thorax diseases

# Import Libraries
from tensorflow import keras
import tensorflow as tf
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Define the path to the CSV file for the bounding box and read the file.
file_path_bblist = r'/YOURPATH/BBox_List_2017.csv'
df_bblist = pd.read_csv(file_path_bblist)
object_names = np.unique(df_bblist["Finding Label"])
object_names = list(object_names)

# Extract image names from the first column "Image Index".
image_names = df_bblist["Image Index"].tolist()

# Output the number of extracted images.
print(f"Total number of images:", len(image_names)) # 984 images available.
print(image_names[:5]) # Output the first 5 image names.

# Information about the number of duplicates within the 984 extracted images.
# The BBox_List_2017.csv file contains multiple images 
# that have been classified with more than one disease (label).

duplicate_values = df_bblist['Image Index'][df_bblist['Image Index'].duplicated(keep=False)].unique()
print(f"Number of images containing multiple diseases:", len(duplicate_values))
print(duplicate_values)

# List for the total number of all images in BBox_List_2017.csv.
counts = df_bblist['Image Index'].value_counts() # Total count of each image.
duplicates = counts[counts>1][:11] # Images that appear more than once in the list.
print(f"Images that appear more than once in the list: \n", duplicates)

# Bar chart for the top 10 most predicted classes.
path = r'/YOURPATH/image_index.png'
plt.figure(figsize=(14,12))
plt.bar(duplicates.index, duplicates.values, color='gray')
plt.title('The 11 most common images with different labels')
plt.xlabel('Image Index')
plt.xticks(rotation=315)
plt.ylabel('Count')
plt.savefig(path, dpi=300)
plt.show()

# Listing of the classes/diseases in the BBox list.
pathologies = df_bblist['Finding Label'].value_counts()
print(f"Number of labels for each disease\n",pathologies)

# Create a bar chart showing the number of diseases.
path = r'/YOURPATH/counts_labels.png'
plt.figure(figsize=(14,12))
plt.bar(pathologies.index, pathologies.values, color='skyblue')
plt.title('Label frequencies')
plt.xlabel('Finding Label')
plt.xticks(rotation=315)
plt.ylabel('Count')
plt.savefig(path, dpi=300)
plt.show()

#%%

# Search for images from the bounding box list within all images using os.walk(),
# then copy these images from the original path to a new folder.
# Finally, save the corresponding bounding box values for each image
# as a new text file in the same new folder.

print("Path of the images with bounding box information:")
dataset_path = r'/YOURPATH/Datensatz'
destination = r'/YOURPATH/BoundingBox'

for dirpath, dirnames, filenames in os.walk(dataset_path):
    for filename in filenames: 
        if filename in image_names:
            full_path = os.path.join(dirpath, filename)
            print(full_path)
         
            # Copy images to the new path.
            destination_path = os.path.join(destination, filename)
            shutil.copy(full_path, destination_path)
         
            # Assign BBox coordinates and labels as an instance over all rows.
            
            bbox_rows = df_bblist.loc[df_bblist["Image Index"] == filename] # All rows for the image.
            
            for index, row in bbox_rows.iterrows(): # Iterate through all bounding boxes for an image.
                bbox_data = row[df_bblist.columns[2:6]].values # BBox coordinates across the rows.
                helper = row[df_bblist.columns[1]] # Labels across all rows.
                pathology = object_names.index(helper)

                # Merge all values in the target folder into a single TXT file.
                if bbox_data.size > 0:  # If bounding box data is available (should always be the case).
                # Scale the bounding box and save it as a text file.
                    bbox_data = bbox_data / 1024
                    # Reshape to ensure it is a single row.
                    bbox_data = np.concatenate(([[pathology]], bbox_data.reshape(1, -1)), axis = 1)
                    # Filename for .txt file
                    txt_file = filename.replace(".png", ".txt")
                    bbox_path = os.path.join(destination, txt_file)

                # Save bounding box coordinates as a text file (append mode).
                if os.path.exists(bbox_path):  # Check if the file already exists.
                    with open(bbox_path, 'a') as f:  # Append-Mode
                        np.savetxt(f, bbox_data, fmt="%.6f")
                else:
                    np.savetxt(bbox_path, bbox_data, fmt="%.6f")  # Create if it does not exist.
                
#%%
from tensorflow import keras
import tensorflow as tf
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
    
# Load images from directory
dataset_path = r'/YOURPATH/Dataset'
df = keras.utils.image_dataset_from_directory(dataset_path,
    labels = None,
    label_mode = "int",
    image_size=(1024,1024),
    batch_size = 1,
    seed=42)

# Display the first 12 images from the dataset.
plt.figure(figsize=(12,12))

for i, image_batch in enumerate(df.take(12)):
    image= tf.squeeze(image_batch).numpy().astype("uint8")
    plt.subplot(3, 4, i + 1)
    plt.imshow(image)
    plt.axis("off")
plt.tight_layout()
plt.show()

#%%

# First training approach for imgsz 128x128

from ultralytics import YOLO
import ultralytics

# load a pretrained model (recommended for training)
model = YOLO("yolov8n.pt")

# Display model information
model.info()

# Train the model
results = model.train(data="/YOURPATH/yolo_xray.yaml",
                      epochs=100, imgsz=128, batch=16, save=True, save_period=10,
                      project='X_Ray_Training', name='YOLOV8_Training', plots=True)

#%%
from ultralytics import YOLO
import ultralytics

# Prediction on 4999 images with imgsz 128x128.
# Load the best model from training for YOLOv8n.
model = YOLO("/YOURPATH/X_Ray_Training/YOLOV8_Training128/weights/best.pt")
result = model.predict(source="/YOURPATH/images001",
                       show=False, save_txt=True, show_labels=True,
                       show_conf=True, show_boxes=True,
                       project='X_Ray_Prediction', name='YOLOV8_Prediction',
                       save=True,
                       imgsz=128,
                       conf=0.25,  # Threshold 1: Keep only objects with the highest probability above this value.
                       iou=0.7,  # Threshold 2: Remove objects that overlap with IOU greater than this value.
                       )

#%%
# Second training approach for imgsz 300x300
from ultralytics import YOLO
import ultralytics

# load a pretrained model (recommended for training)
model = YOLO("yolov8n.pt")

# Display model information
model.info()

# Train the model with higher resolution pictures 300x300
results300 = model.train(data="/YOURPATH/yolo_xray.yaml",
                      epochs=100, imgsz=300, batch=16, save=True, save_period=10,
                      project='X_Ray_Training_300', name='YOLOV8_Training_300', plots=True)

#%%

from ultralytics import YOLO
import ultralytics

# Use the model with higher resolution training (300x300) for prediction.
model = YOLO("/YOURPATH/X_Ray_Training_300/YOLOV8_Training_300/weights/best.pt")

# Prediction on 4999 images with imgsz 300x300.
# Load the best model from training for YOLOv8n.
results300pred = model.predict(source="/YOURPATH/images001",
                       show=False, save_txt=True, show_labels=True,
                       show_conf=True, show_boxes=True,
                       project='X_Ray_Prediction_300', name='YOLOV8_Prediction_001',
                       save=True,
                       imgsz=300,
                       conf=0.25,  # Threshold 1: Keep only objects with the highest probability above this value.
                       iou=0.7,  # Threshold 2: Remove objects that overlap with IOU greater than this value.
                       )

