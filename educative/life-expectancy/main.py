from pathlib import Path
import numpy as np

# computing the predictions
def predict(X, w):
    return np.matmul(X, w)

# calculating the loss
def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)

# evaluating the gradient
def gradient(X, Y, w):
    return 2 * np.matmul(X.T, (predict(X, w) - Y)) / X.shape[0]

# performing the training phase for our classifier
def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        print("Iteration %4d => Loss: %.20f" % (i, loss(X, Y, w)))
        w -= gradient(X, Y, w) * lr
    return w

# Import the dataset
data_path = Path(__file__).parent / "life-expectancy-without-country-names.txt"
if not data_path.exists():
    raise FileNotFoundError(
        f"{data_path} not found. Put pizza.txt next to pizzeria.py or adjust the path."
    )

x1, x2, x3, y = np.loadtxt(data_path, skiprows=1, unpack=True)
X = np.column_stack((np.ones(x1.size), x1, x2, x3))
Y = y.reshape(-1, 1)
w = train(X, Y, iterations=5000000, lr=0.0001)

print("\nWeights: %s" % w.T)
print("\nA few predictions:")
for i in range(10):
    print("X[%d] -> %.4f (label: %d)" % (i, predict(X[i], w), Y[i]))