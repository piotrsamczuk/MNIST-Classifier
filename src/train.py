import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from data_loader import load_data
from model import create_model

def train_model(epochs):
    (x_train, y_train), (x_test, y_test) = load_data()
    model = create_model()
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=1)
    os.makedirs('models', exist_ok=True)
    model.save('models/mnist_model.keras')
    os.makedirs('metrics', exist_ok=True)
    generate_confusion_matrix(model, x_test, y_test)
    plot_performance_metrics(history, epochs)

def generate_confusion_matrix(model, x_test, y_test):
    predictions = model.predict(x_test)
    y_pred = np.argmax(predictions, axis=1)
    cm = tf.math.confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig('metrics/confusion_matrix.png')
    plt.close()
    accuracy = tf.reduce_sum(tf.cast(tf.equal(y_test, y_pred), tf.float32)) / len(y_test)
    with open('metrics/classification_report.txt', 'w') as file:
        file.write(f'Test Accuracy: {accuracy.numpy()}')

def plot_performance_metrics(history, epochs):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history.history['accuracy'], label='Training Accuracy')
    plt.plot(range(1, epochs + 1), history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Epoch vs. Performance (Accuracy)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history.history['loss'], label='Training Loss')
    plt.plot(range(1, epochs + 1), history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs. Performance (Loss)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('metrics/epoch_vs_performance.png')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on MNIST dataset.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train the model for.')
    args = parser.parse_args()

    train_model(args.epochs)
