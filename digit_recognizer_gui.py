import numpy as np
import tkinter as tk


class DigitRecognizer:
    def __init__(self, root, model):
        self.root = root
        self.model = model

        # Creation of the main canvas for drawing
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()

        # Creating an invisible canvas for 28x28 pixels
        self.hidden_canvas = np.zeros((28, 28), dtype=np.float32)

        # Connection of mouse movements with the drawing method
        self.canvas.bind("<B1-Motion>", self.draw)

        # Buttons to predict and clear
        self.predict_button = tk.Button(root, text="Recognize", command=self.predict_digit)
        self.predict_button.pack()
        self.clear_button = tk.Button(root, text="Clean up", command=self.clear_canvas)
        self.clear_button.pack()

    def draw(self, event):
        # Drawing on the main canvas
        x, y = event.x, event.y
        size = 10  # Brush size
        self.canvas.create_oval(x - size, y - size, x + size, y + size, fill='black')

        # Drawing on a 28x28 invisible layer
        hidden_x, hidden_y = int(x / 10), int(y / 10)  # Coordinate scaling
        if 0 <= hidden_x < 28 and 0 <= hidden_y < 28:
            self.hidden_canvas[hidden_y, hidden_x] = 1.0  # Filling the pixel

    def clear_canvas(self):
        # Clean both canvases
        self.canvas.delete("all")
        self.hidden_canvas.fill(0)

    def predict_digit(self):
        # The hidden_canvas array already has the required size of 28x28
        image_array = self.hidden_canvas.reshape(1, 28, 28, 1)  # Format for model [1, 28, 28, 1]

        # Making predictions
        predicted_class = np.argmax(self.model.predict(image_array))  # Prediction of the class
        print(f'Estimated number: {predicted_class}')
