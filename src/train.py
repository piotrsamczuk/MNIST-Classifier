import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from data_loader import load_data
from model import create_model

class PerIterationPlotter(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []
        os.makedirs('metrics', exist_ok=True)
        
    def on_batch_end(self, batch, logs=None):
        self.train_acc.append(logs['accuracy'])
        self.train_loss.append(logs['loss'])
        
    def on_test_batch_end(self, batch, logs=None):
        if logs:
            self.val_acc.append(logs['accuracy'])
            self.val_loss.append(logs['loss'])
            
    def on_train_end(self, logs=None):
        self._save_plots()
            
    def _save_plots(self):
        iterations = range(1, len(self.train_acc) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.plot(iterations, self.train_acc, label='Training Accuracy')
        if self.val_acc:
            ax1.plot(iterations[-len(self.val_acc):], self.val_acc, label='Validation Accuracy')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Iteration vs. Performance (Accuracy)')
        ax1.legend()
        
        ax2.plot(iterations, self.train_loss, label='Training Loss')
        if self.val_loss:
            ax2.plot(iterations[-len(self.val_loss):], self.val_loss, label='Validation Loss')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.set_title('Iteration vs. Performance (Loss)')
        ax2.legend()
        
        plt.tight_layout()
        fig.savefig('metrics/iteration_vs_performance.png')
        plt.close()

def train_model(epochs):
    (x_train, y_train), (x_test, y_test) = load_data()
    model = create_model()
    plotter = PerIterationPlotter()
    history = model.fit(x_train, y_train, epochs=epochs, 
                       validation_data=(x_test, y_test), 
                       callbacks=[plotter], 
                       verbose=1)
    
    os.makedirs('models', exist_ok=True)
    model.save('models/mnist_model.keras')
    
    generate_confusion_matrix(model, x_test, y_test)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on MNIST dataset.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train the model for.')
    args = parser.parse_args()
    
    train_model(args.epochs)