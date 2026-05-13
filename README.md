# Snow Detection System

## Overview

This project is an image-based snow detection system developed to estimate snow-covered conditions from road or campus images.

The system combines semantic segmentation and image classification:

1. **U²-Net** is used to extract the valid road/background area from the input image.
2. The extracted area is divided into small grid images.
3. **EfficientNetV2** is used to classify each grid and estimate the snow-covered condition.
4. The final result is saved as a CSV file.

By combining segmentation and classification, the system can not only determine whether snow exists, but also estimate the degree of snow coverage more precisely.

---

## System Flow

```text
Input Image
    ↓
U²-Net Semantic Segmentation
    ↓
Road / Background Area Extraction
    ↓
64 × 64 Grid Division
    ↓
EfficientNetV2 Classification
    ↓
Snow Coverage Estimation
    ↓
CSV Output
```

---

## Models

### U²-Net: Semantic Segmentation Model

U²-Net is used to extract the target area from the input image.

- **Input:** original road or campus image
- **Output:** segmentation mask
- **Purpose:** remove unnecessary areas and keep the region used for snow judgment

In this project, the segmentation result is used as a mask, and the extracted image is passed to the next classification step.

### EfficientNetV2: Image Classification Model

EfficientNetV2 is used to classify small grid images after segmentation.

- **Input:** 64 × 64 grid images extracted from the segmented image
- **Output:** probability of snow-covered condition
- **Classes:** snow / non-snow

The output probability of the binary classification model is used approximately as the snow coverage score for each small grid image.

---

## Project Structure

```text
challenge/exam/
│
├── model/
│   ├── mode.py                  # U²-Net model definition
│   └── efficient_netv2.py       # EfficientNetV2 model definition
│
├── result/
│   └── result.csv               # Final output result
│
├── Mydataset.py                 # Dataset loader for U²-Net
├── Mydataset_efficient.py       # Dataset loader for EfficientNetV2
├── train_u2net.py               # Training script for U²-Net
├── Test_u2net.py                # Test script for U²-Net segmentation
├── train_test_efficientnet.py   # Training script for EfficientNetV2
├── class_test.py                # Classification test script
├── final_test.py                # Integrated final test script
├── early_stop.py                # Early stopping utility
└── data_loader.py               # Data loading and preprocessing utilities
```

---

## Dataset Structure

The dataset path used in the code is as follows:

```text
data/
├── u2_snow_data/
│   ├── train/
│   │   ├── imgs/
│   │   └── mask/
│   └── test/
│       ├── imgs/
│       └── mask/
│
└── classify_data_ef/
    ├── train/
    └── test/
```

---

## Environment

- Python 3.9+
- PyTorch
- torchvision
- OpenCV
- NumPy
- SciPy
- TensorBoard

Install required packages:

```bash
pip install torch torchvision opencv-python numpy scipy tensorboard
```

---

## Training

### Train U²-Net

```bash
python train_u2net.py
```

The trained model will be saved under:

```text
model/save/
```

### Train EfficientNetV2

```bash
python train_test_efficientnet.py
```

The best classification model will be saved under:

```text
model/save/
```

---

## Testing

### Test U²-Net Segmentation

```bash
python Test_u2net.py
```

This script evaluates the segmentation model and calculates the Dice coefficient.

### Run Final Integrated Test

```bash
python final_test.py
```

The final result will be saved as:

```text
result/result.csv
```

The CSV file contains:

```text
number, img_path, time_cost, pre_value
```

| Column | Description |
|---|---|
| number | Image index |
| img_path | Image file name |
| time_cost | Total processing time |
| pre_value | Estimated snow coverage score |

---

## Main Features

- Semantic segmentation using U²-Net
- Image classification using EfficientNetV2
- Grid-based snow coverage estimation
- Automatic CSV result generation
- PyTorch-based implementation

---

## Method

### Step 1: Semantic Segmentation

First, U²-Net extracts the road or ground region from the original image.

This step removes unnecessary background information such as buildings or sky and keeps the region related to snow detection.

### Step 2: Grid Division

The segmented image is divided into multiple small images of size 64 × 64.

This method enables more detailed local snow analysis.

### Step 3: Classification

Each grid image is classified using EfficientNetV2.

The model outputs the probability of snow existence.

### Step 4: Snow Coverage Estimation

The classification probabilities are approximately used as snow coverage values.

Finally, the overall snow-covered condition is estimated from all grid results.

---

## Purpose of the Project

This project was developed to support safer campus or road management during winter.

By automatically estimating the snow-covered condition from camera images, the system can help administrators decide whether snow removal is necessary.

For example, if the estimated snow coverage exceeds a certain threshold, the system can be connected to a notification function to alert administrators.

---

## Future Work

- Improve model accuracy with more training data
- Add real-time camera input
- Build an automatic notification system
- Deploy the model on a cloud server
- Improve the threshold setting for snow removal judgment

---

## Author

Shijie Yang
