import numpy as np
from pathlib import Path


def predict(X, w, b):
    return X * w + b


def gradient(X, Y, w):
    return 2 * np.average(X * (predict(X, w, 0) - Y))


def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)


def train(X, Y, iterations, lr):
    w = 0
    for i in range(iterations):
        print("Iteration %4d => Loss: %.10f" % (i, loss(X, Y, w, 0)))
        w -= gradient(X, Y, w) * lr
    return w


# Import the dataset
data_path = Path(__file__).parent / "pizza.txt"
if not data_path.exists():
    raise FileNotFoundError(
        f"{data_path} not found. Put pizza.txt next to pizzeria.py or adjust the path."
    )

X, Y = np.loadtxt(data_path, skiprows=1, unpack=True)
w = train(X, Y, iterations=100, lr=0.001)
print("\nw=%.10f" % w)
