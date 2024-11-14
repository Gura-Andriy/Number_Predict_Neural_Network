from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Dropout
import numpy as np
import os


class MNISTModel:
    def __init__(self, model_path='mnist_model.h5'):
        self.model_path = model_path
        self.model = None

    def load_data(self):
        # Loading and normalizing MNIST data
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalizing
        y_train = to_categorical(y_train, 10)  # One-hot encoding labels
        y_test = to_categorical(y_test, 10)
        return x_train, y_train, x_test, y_test

    def create_model(self):
        # Create a new model with Dropout
        self.model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation='relu'),
            Dropout(0.2),  # Add Dropout with a probability of 20%
            Dense(10, activation='softmax')
        ])
        # Compile the model
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        print("The new model is made with Dropout.")

    def load_or_train_model(self, x_train, y_train, epochs=5, batch_size=32):
        # Checking the presence of a saved model
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            print("The saved model is loaded.")
        else:
            # Create and train a model if it does not exist
            self.create_model()
            self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
            self.save_model()
            print("The model is trained and saved.")

    def save_model(self):
        # Saving the model to a file
        if self.model:
            self.model.save(self.model_path)
            print(f"The model is saved in the file: {self.model_path}")

    def evaluate_model(self, x_test, y_test):
        # Accuracy assessment on test data
        if self.model:
            test_loss, test_accuracy = self.model.evaluate(x_test, y_test)
            print(f'Accuracy on test data: {test_accuracy * 100:.2f}%')
        else:
            print("The model is not loaded.")

    def predict(self, image):
        # Prediction for one image
        if self.model:
            image = image.reshape(1, 28, 28)  # Conversion to the required format
            predicted_class = np.argmax(self.model.predict(image), axis=-1)
            return predicted_class[0]
        else:
            print("The model is not loaded.")
            return None
