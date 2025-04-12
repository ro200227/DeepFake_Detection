# Deepfake Detection: Image and Audio using Deep Learning (CNN) & Librosa

This repository contains a deep learning project for detecting deepfake **audio** and **image** files using **Convolutional Neural Networks (CNN)**. The project leverages **Librosa** for audio processing (MFCC features extraction) and deep learning-based CNN models for both **image** and **audio** deepfake detection. The model is deployed as an interactive web application using **Streamlit**.

## Project Overview

This project is designed to detect deepfake media in the form of **images** and **audio**. It uses **Deep Learning** with **CNNs (Convolutional Neural Networks)** to classify images and audio files as real or deepfake. For **audio files**, **Librosa** is used to extract **MFCC (Mel Frequency Cepstral Coefficients)**, which are then used to classify the audio content as real or deepfake.

### Key Features:
- **Image Detection**: Uses **Deep Learning (CNN)** for image-based deepfake detection.
- **Audio Detection**: Uses **Librosa** for MFCC feature extraction and **Deep Learning (CNN)** for deepfake audio classification.
- **Web Application**: The model is deployed using **Streamlit**, allowing easy interaction with the model via a simple web interface.

## Installation

### Prerequisites

To run this project, you need **Python 3.x** and the following dependencies:

- **PyTorch**: For building and training the CNN models for both image and audio classification.
- **Librosa**: For audio processing and MFCC feature extraction.
- **NumPy**, **Pandas**, **Scikit-learn**: For data handling and machine learning tasks.
- **Streamlit**: For deploying the model as a web application.

