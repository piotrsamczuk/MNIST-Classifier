# MNIST Classifier Project

## Overview
This project implements a classifier for the MNIST dataset, which consists of 70,000 grayscale images of handwritten digits (0-9). The classifier is built using Python and utilizes deep learning techniques to achieve high accuracy in digit recognition. It also includes a drawing interface to recognize handwritten digits.

## Features
- Preprocessing of MNIST dataset
- Building and training a neural network model using TensorFlow/Keras
- Evaluating the model's performance
- Visualizing predictions
- Drawing interface to predict handwritten digits

## Usage

Optionally create a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate # On Windows, use `venv\Scripts\activate`
    ```

1. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

2. **Train the model:**
    ```sh
    python3 src/train.py
    ```

    To train the model for a specific number of epochs, use the --epochs flag:
    ```sh
    python3 src/train.py --epochs <number_of_epochs>
    ```

3. **Evaluate the model:**
    ```sh
    python3 src/evaluate.py
    ```

4. **Visualize predictions:**
    ```sh
    python3 src/visualize.py
    ```

5. **Draw and predict digits:**
    ```sh
    python3 src/draw_and_predict.py
    ```