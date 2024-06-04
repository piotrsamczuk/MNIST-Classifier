# src/draw_and_predict.py

import tkinter as tk
from tkinter import messagebox
import os
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a digit")
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()
        
        self.model = load_model('models/mnist_model.keras')
        
        self.button_predict = tk.Button(root, text="Predict", command=self.predict_digit)
        self.button_predict.pack()
        
        self.button_clear = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()
        
        self.canvas.bind("<B1-Motion>", self.draw)
        
        self.image = Image.new("L", (280, 280), 255)
        self.draw_img = ImageDraw.Draw(self.image)
        
    def draw(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
        self.draw_img.ellipse([x - r, y - r, x + r, y + r], fill='black')
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)
        self.draw_img = ImageDraw.Draw(self.image)
    
    def preprocess_image(self):
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)

        # Find the bounding box of the non-zero pixels
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        
        # Resize the cropped image to maintain aspect ratio
        img = img.resize((20, 20), Image.LANCZOS)

        # Create a new 28x28 white image
        new_img = Image.new("L", (28, 28), 0)
        
        # Calculate the position to paste the centered digit
        paste_pos = ((28 - img.size[0]) // 2, (28 - img.size[1]) // 2)
        new_img.paste(img, paste_pos)
        
        new_img = np.array(new_img).astype('float32') / 255.0
        new_img = new_img.reshape(1, 28, 28, 1)
        return new_img
    
    def predict_digit(self):
        img = self.preprocess_image()
        prediction = self.model.predict(img)
        digit = np.argmax(prediction)
        
        plt.figure(figsize=(2, 2))
        plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
        plt.title(f"Predicted Digit: {digit}")
        plt.show()

if __name__ == "__main__":
    model_file = 'models/mnist_model.keras'
    if not os.path.isfile(model_file):
        messagebox.showerror("Error", "Model file not found. Please generate the model first.")
    else:
        root = tk.Tk()
        app = App(root)
        root.mainloop()
