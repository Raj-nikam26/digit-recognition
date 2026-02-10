# Handwritten-Digit-Recognition-CNN-Flask-App

This project demonstrates a Convolutional Neural Network model created using PyTorch library over the MNIST dataset to recognize handwritten digits . 


# Project Overview

1. Data ingestion and preprocessing
2. CNN model training using PyTorch
3. Model evaluation and checkpointing
4. Inference-only model loading
5. Model serving logic
6. Interactive Streamlit UI for predictions

# Dataset

The MNIST dataset is an acronym that stands for the Modified National Institute of Standards and Technology dataset.

It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

Images are flattened into 784 pixel values

The training set includes labels

## Download Dataset

Kaggle – Digit Recognizer (MNIST):
https://www.kaggle.com/competitions/digit-recognizer/data

After downloading, place the files here:
```
data/train.csv
data/test.csv
```
# Model Architecture

Custom Convolutional Neural Network (CNN)
Multiple convolution + pooling layers
Dropout for regularization
Fully connected layers for classification
LogSoftmax output with NLLLoss
The model predicts one of 10 digit classes (0–9).

# Technologies Used

Python
PyTorch
NumPy
Pandas
Matplotlib
Streamlit


# Train the model

```
python train.py
```
# Running Inference 

```
python app.py
```


