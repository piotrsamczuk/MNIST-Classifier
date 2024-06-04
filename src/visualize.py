import matplotlib.pyplot as plt
import numpy as np
import os
from tkinter import messagebox
from tensorflow.keras.models import load_model
from data_loader import load_data

def visualize_predictions():
    (x_train, y_train), (x_test, y_test) = load_data()

    model_file = 'models/mnist_model.keras'
    if not os.path.isfile(model_file):
        messagebox.showerror("Error", "Model file not found. Please generate the model first.")
        return
    else:
        model = load_model(model_file)

    predictions = model.predict(x_test)

    num_images = 10
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.binary)
        plt.xlabel(np.argmax(predictions[i]))
    plt.show()

if __name__ == "__main__":
    visualize_predictions()