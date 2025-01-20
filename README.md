# Lung Cancer Image Classification

This project is designed to classify lung cancer images into two categories: **Malignant** and **Normal** using a Convolutional Neural Network (CNN). The model is trained on grayscale images and provides a prediction with a probability of lung cancer presence.

## Table of Contents
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Training](#model-training)
- [Model Prediction](#model-prediction)
- [Visualizations](#visualizations)

---

## Project Overview
This program utilizes TensorFlow to build a Convolutional Neural Network (CNN) model for binary classification of lung cancer images. The model is trained using a dataset of lung images stored in a directory, and predictions are made on new images using a trained model. The program also includes visualizations of training accuracy and loss.

---

## Requirements

Before running the program, ensure you have the following installed:

- Python 3.x
- TensorFlow (>= 2.x)
- OpenCV
- NumPy
- Matplotlib

You can install the necessary libraries using the following pip command:

```
pip install tensorflow opencv-python numpy matplotlib
```

## Usage

### 1. **Running the Model on New Image**

After the model is trained, you can use the script to make predictions on new images.

#### Command Line Syntax:
```
python predict.py <image-path>
```

This will output one of the following based on the model's prediction:
- "This patient likely has lung cancer"
- "This patient likely does NOT have lung cancer"
- "Unsure whether this patient has lung cancer or not"

Additionally, it will print the percentage likelihood of having lung cancer, with values representing the probability that the image is classified as **Malignant** or **Normal**.