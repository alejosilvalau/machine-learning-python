import numpy as np
from pathlib import Path

# Import the dataset
data_path = Path(__file__).parent / "pizza_3_vars.txt"
if not data_path.exists():
    raise FileNotFoundError(
        f"{data_path} not found. Put pizza.txt next to pizzeria.py or adjust the path."
    )

# Import the dataset
x1, x2, x3, y = np.loadtxt(data_path, skiprows=1, unpack=True)

print(x1.shape)

X = np.column_stack((x1, x2, x3))
print(X.shape)

print(X[:2])

Y = y.reshape(-1, 1)
print(Y.shape)

w = np.zeros((X.shape[1], 1))
w.shape  # => (3, 1)`


# w.shape[0] == number of rows, w.shape[1] == number of columns
def predict(X, w):
    return np.matmul(X, w)


def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)


a_number = loss(X, Y, w)
a_number.shape  # => ()


def gradient(X, Y, w):
    return 2 * np.matmul(X.T, (predict(X, w) - Y)) / X.shape[0]


# X.T means that the X was transposed
# we are not using the average function here, but we are performing the average manually
