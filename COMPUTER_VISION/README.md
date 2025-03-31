# Computer Vision for Melanoma Detection

This folder contains various scripts and resources focused on the detection of melanomas in images. The goal is to develop and improve computer vision models to classify images as benign or malignant melanomas.

## Contents

- **`modelo_1.ipynb`**: 
  A simple convolutional neural network (CNN) model that extracts features from images to classify whether a melanoma is present or not. This script serves as a starting point for melanoma detection.

- **`modelo_1_torch.ipynb`**: 
  An optimized version of the model designed to run on a GPU using CUDA, addressing the high computational cost of the previous script. This script leverages a pre-trained ResNet model for feature extraction and fine-tuning.

- **`melanoma_model_1_torch.pth`**: 
  The saved weights of the model trained in `modelo_1_torch.ipynb`.

- **`melanoma_model_1.weights.h5`**:  
  The saved weights of the model trained in `modelo_1.ipynb`.

- **`preprocessing_dataISIC.ipynb`**:  
  A script for preprocessing the ISIC dataset, including resizing, normalization, and data augmentation techniques.

- **`loading_dataset_kaddle.ipynb`**:  
  A script for loading and preparing datasets from Kaggle for training and evaluation.

## Dataset

The models in this folder have been trained using a melanoma dataset obtained from Kaggle. This dataset contains labeled images of skin lesions, which were used to train and evaluate the performance of the models.


## Requirements


- For `modelo_1_torch.ipynb`, ensure that CUDA is installed and a compatible GPU is available to run the script efficiently.
- Pre-trained ResNet weights are automatically downloaded when running the script.

## Purpose

These scripts aim to explore different approaches to melanoma detection, starting from basic CNN models to more advanced techniques using pre-trained architectures. The folder provides a foundation for further experimentation and improvement in melanoma classification tasks.