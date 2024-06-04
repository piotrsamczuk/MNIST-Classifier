from data_loader import load_data
from tensorflow.keras.models import load_model
import os
from tkinter import messagebox

def evaluate_model():
    (x_train, y_train), (x_test, y_test) = load_data()
    model_file = 'models/mnist_model1.keras'
    if not os.path.isfile(model_file):
        messagebox.showerror("Error", "Model file not found. Please generate the model first.")
        return
    else:
        model = load_model(model_file)
        
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc}")

if __name__ == "__main__":
    evaluate_model()