import numpy as np
from pathlib import Path


# Calling the predict() function
def predict(X, w, b):
    return X * w + b


# Calculatin the loss
def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)


# computing the derivative
def gradient(X, Y, w, b):
    w_gradient = 2 * np.average(X * (predict(X, w, b) - Y))
    b_gradient = 2 * np.average(predict(X, w, b) - Y)
    return (w_gradient, b_gradient)


# calling the training function for 20,000 iterations
def train(X, Y, iterations, lr):
    w = b = 0
    for i in range(iterations):
        if i % 5000 == 0:
            print("Iteration %4d => Loss: %.10f" % (i, loss(X, Y, w, b)))
        w_gradient, b_gradient = gradient(X, Y, w, b)
        w -= w_gradient * lr
        b -= b_gradient * lr
    return w, b


# Import the dataset
data_path = Path(__file__).parent / "pizza.txt"
if not data_path.exists():
    raise FileNotFoundError(
        f"{data_path} not found. Put pizza.txt next to pizzeria.py or adjust the path."
    )

# loading the data and then calling the desired functions
X, Y = np.loadtxt(data_path, skiprows=1, unpack=True)
w, b = train(X, Y, iterations=20000, lr=0.001)
print("\nw=%.10f, b=%.10f" % (w, b))
print("Prediction: x=%d => y=%.2f" % (20, predict(20, w, b)))
