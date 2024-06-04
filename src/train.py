import os
from data_loader import load_data
from model import create_model

def train_model():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = create_model()
    
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    model.save('models/mnist_model.h5')

if __name__ == "__main__":
    train_model()
