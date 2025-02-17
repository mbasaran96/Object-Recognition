# Object-Detection
Image localization model of common thorax diseases 

This project shows the approach of generating a deep-learning model for the localization of common thoracic diseases. <br />
The work is based on the following scientific paper: <br />
https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf 

The National Institutes of Health published 2017 a new chest X-ray database - "*ChestX-ray8*" - comprising over **100,000** frontal view X-ray images of 32,717 unique patients with the text-mined eight disease labels, where each image can have multiple labels. <br />

The goal of this project is to utilize the dataset and address the challenges of high-precision computer-aided diagnosis systems.

# Dataset

The dataset contains a total of **112,120 images**, each with a size of 1024×1024. <br />
The images are split into 12 zip files, which together comprise a total of 42 GB of storage space. <br />
Out of 112,120 images, more than 24,000 are being labeled with more than one disease. <br />
The dataset can be downloaded via the following link: https://nihcc.app.box.com/v/ChestXray-NIHCC

# Bounding Box

The bounding box list (*BBox_List_2017.csv*) can also be downloaded from the link shared above. <br />
It contains the associated labels with its coordinates for only 984 images. <br />
Out of 984 images, 93 duplicates are included in the list, indicating that these have more than one pathology classified. <br />

# YOLOv8n

To detect multiple objects in one image, the YOLOv8n algorithm is used, relying on fully connected convolutional layers. <br />
The download and implementation of the model can be found on the following website: https://docs.ultralytics.com/de/models/yolov8/#citations-and-acknowledgements

- The *yolo_xray.yaml* file contains the path of the bounding box, which has three subfolders - Train, Val, and Test
- The split for the three subfolders was manually performed to 50 / 25 / 25
- The Classes/names of all diseases have also to be listed in the same file





