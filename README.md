# COVID X-ray Classification Using CNN

This project aims to build a Convolutional Neural Network (CNN) model to classify X-ray images into three categories: **COVID**, **Normal**, and **Viral Pneumonia**. The model is trained using a dataset of X-ray images and evaluated based on its ability to predict these categories accurately. The project uses various tools and libraries such as TensorFlow, Keras, OpenCV, and Scikit-learn.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Dependencies](#dependencies)
4. [Project Structure](#project-structure)
5. [How to Run the Project](#how-to-run-the-project)
6. [Model Architecture](#model-architecture)
7. [Evaluation](#evaluation)
8. [Results](#results)
9. [Future Work](#future-work)
10. [License](#license)

---

## Project Overview

The objective of this project is to train a CNN model for detecting COVID-19, normal, and viral pneumonia based on X-ray images. This project is intended to help healthcare professionals diagnose COVID-19 more accurately and quickly by analyzing X-ray images. The model is trained, evaluated, and deployed to demonstrate real-world application.

---

## Dataset

The dataset used in this project is from the **COVID-19 Chest X-ray Dataset**. It consists of X-ray images labeled as **COVID**, **Normal**, and **Viral Pneumonia**. The images are pre-processed, resized, and normalized before training the model.

- Dataset source: [Kaggle COVID CXR Dataset](https://www.kaggle.com/datasets/sid321axn/covid-cxr-image-dataset-research)

---

## Dependencies

This project requires the following libraries:

- **TensorFlow**: Deep learning framework used for model building and training.
- **Keras**: High-level API for neural networks.
- **OpenCV**: Used for image preprocessing.
- **Matplotlib**: For visualizing results.
- **Scikit-learn**: For model evaluation and splitting the dataset.

You can install the required dependencies by running:

```bash
pip install -r requirements.txt
