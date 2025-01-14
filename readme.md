# ML Iris Classification Project

This project demonstrates simple machine learning models for classifying Iris flowers using scikit-learn.

## Project Structure
mlops-sample-project/
│
├── train.py
├── tests/
│   └── test_sample.py
└── README.md

## Description

This project uses the Iris dataset to train and compare two machine learning models:
1. Random Forest Classifier
2. Logistic Regression

The `train.py` script loads the data, splits it into training and testing sets, trains both models, and evaluates their performance.

## Installation

To run this project, you need Python 3.6+ and the following libraries:
- scikit-learn
- numpy

You can install the required packages using pip:
pip install scikit-learn numpy

## Usage

To train the models and see the results, run:
python train.py

This will output the accuracy and classification reports for both models.

## Testing

To run the tests, make sure you have pytest installed:
pip install pytest
