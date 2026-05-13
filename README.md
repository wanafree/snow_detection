# Snow Detection System

## Overview

This project is a snow detection system based on image analysis.  
The system aims to detect snow-covered areas from road or campus images and estimate the degree of snow coverage.

The main idea of this project is to combine:

1. **Semantic segmentation** using U²-Net
2. **Image classification** using EfficientNetV2
3. **Grid-based snow coverage estimation**

By using segmentation and classification together, the system can not only judge whether snow exists, but also estimate the snow-covered condition more precisely.

---

## Project Structure

```bash
challenge/exam/
│
├── model/
│   ├── mode.py                 # U²-Net model definition
│   └── efficient_netv2.py       # EfficientNetV2 model definition
│
├── result/
│   └── result.csv              # Output result file
│
├── Mydataset.py                # Dataset loader for U²-Net segmentation
├── Mydataset_efficient.py      # Dataset loader for EfficientNetV2 classification
├── train_u2net.py              # Training script for U²-Net
├── Test_u2net.py               # Test script for U²-Net segmentation
├── train_test_efficientnet.py  # Training script for EfficientNetV2
├── class_test.py               # Test script for classification
├── final_test.py               # Final integrated test script
├── early_stop.py               # Early stopping utility
└── data_loader.py              # Data loading and preprocessing utilities
