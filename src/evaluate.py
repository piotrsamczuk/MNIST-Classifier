from data_loader import load_data
from tensorflow.keras.models import load_model

def evaluate_model():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = load_model('models/mnist_model.h5')
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc}")

if __name__ == "__main__":
    evaluate_model()