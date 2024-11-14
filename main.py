import tkinter as tk
from tensorflow.keras.models import load_model
from digit_recognizer_gui import DigitRecognizer

# Loading the neural network model
model = load_model('mnist_model.h5')

# Initialize the Tkinter main window
root = tk.Tk()
root.title("Digit Recognizer")

# Create an instance of the DigitRecognizer class
app = DigitRecognizer(root, model)

# Start Tkinter main loop
root.mainloop()