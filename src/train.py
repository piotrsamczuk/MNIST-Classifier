import os
import argparse
from data_loader import load_data
from model import create_model

def train_model(epochs):
  (x_train, y_train), (x_test, y_test) = load_data()
  model = create_model()
  
  model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
  
  # Save the model
  os.makedirs('models', exist_ok=True)
  model.save('models/mnist_model.keras')

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train a model on MNIST dataset.')
  parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train the model for.')
  args = parser.parse_args()

  train_model(args.epochs)
