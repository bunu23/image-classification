## Convolutional Neural Network

# Multi-Class Image Classification with Transfer Learning

This repository contains a Jupyter notebook for building a multi-class image classifier using transfer learning with a pre-trained ResNet-50 model. The notebook covers various steps including data preprocessing, model selection, training, evaluation, fine-tuning, and performance analysis.

## Overview

Image classification is a fundamental task in computer vision, where the goal is to assign a label or category to an input image. Transfer learning is a popular technique in deep learning, where pre-trained models developed for a task are reused as the starting point for a new task. In this notebook, transfer learning is utilized to build an image classifier for a multi-class dataset.

## Dataset

The dataset used in this project consists of images belonging to multiple classes. It is split into training, validation, and test sets. The images are preprocessed using a custom `ImageDataGenerator` to handle transparency and resized to a standard size.

## Model Architecture

The ResNet-50 architecture is used as the base model, which is a deep convolutional neural network that has shown strong performance on various image classification tasks. Custom layers are added on top of the ResNet-50 base for classification. The model is compiled with the Adam optimizer, categorical cross-entropy loss function, and accuracy as the evaluation metric.

## Training and Evaluation

The model is trained on the training data with validation on the validation data. The trained model is then evaluated on the test data to measure its performance in terms of accuracy. Additionally, the training history is analyzed by plotting accuracy and loss curves.

## Fine-Tuning and Optimization

To further improve the model's performance, some layers of the base model are unfrozen for fine-tuning. The model is recompiled with a lower learning rate and retrained on the data. The fine-tuned model is then evaluated on the test data to assess its improved accuracy.

## Sample Predictions

Sample predictions are visualized on the test data to understand how well the model performs on unseen images. Additionally, a function is provided to predict the class of an external image using the trained model.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Matplotlib
- NumPy
- PIL
