# MNIST Digit Recognition

## Overview
A classic machine learning project for handwritten digit recognition using the MNIST dataset. This project demonstrates fundamental computer vision and deep learning skills.

## Project Description
The MNIST dataset contains 70,000 grayscale images of handwritten digits (0-9), each 28x28 pixels. This project implements neural networks to classify these digits with high accuracy.

## Features
- **Data Loading & Preprocessing**: Normalization and reshaping of image data
- **Multiple Model Architectures**:
  - Convolutional Neural Network (CNN) with TensorFlow/Keras
  - Multi-Layer Perceptron (MLP) with scikit-learn (fallback)
- **Comprehensive Evaluation**:
  - Accuracy metrics
  - Confusion matrix
  - Classification report
- **Visualizations**:
  - Sample digits from dataset
  - Training history (accuracy/loss curves)
  - Misclassified examples
  - Confusion matrix heatmap

## Results
- **CNN Accuracy**: ~99%+ on test set
- **MLP Accuracy**: ~97%+ on test set

## How to Run
```bash
# Clone repository
git clone https://github.com/yourusername/mnist-digit-recognizer.git
cd mnist-digit-recognizer

# Install dependencies
pip install -r requirements.txt

# Run the classifier
python mnist_classifier.py
