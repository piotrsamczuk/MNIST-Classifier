import os
import argparse
from data_loader import load_data
from tensorflow.keras.models import load_model
from tkinter import messagebox

def evaluate_model(filename):
    (x_train, y_train), (x_test, y_test) = load_data()
    model_file = f'models/{filename}.keras'
    if not os.path.isfile(model_file):
        messagebox.showerror("Error", f"Model file '{filename}' not found. Please generate the model first.")
        return
    else:
        model = load_model(model_file)
        
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a model on MNIST dataset.')
    parser.add_argument('--filename', type=str, default='mnist_model', help='Name of the model file to be evaluated.')
    args = parser.parse_args()

    evaluate_model(args.filename)
