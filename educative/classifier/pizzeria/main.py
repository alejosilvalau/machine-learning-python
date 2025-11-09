# A binary classifier.

from pathlib import Path
import numpy as np


# Applying Logistic Regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Basically doing prediction but named forward as its
# performing Forward-Propagation
def forward(X, w):
    weighted_sum = np.matmul(X, w)
    return sigmoid(weighted_sum)


# Calling the predict() function
def classify(X, w):
    return np.round(forward(X, w))


# Computing Loss over using logistic regression
def loss(X, Y, w):
    y_hat = forward(X, w)
    first_term = Y * np.log(y_hat)
    second_term = (1 - Y) * np.log(1 - y_hat)
    return -np.average(first_term + second_term)


# calculating gradient
def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]


# calling the training function for 10,000 iterations
def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        if i % 2000 == 0 or i == 9999:
            print("Iteration %4d => Loss: %.20f" % (i, loss(X, Y, w)))
        w -= gradient(X, Y, w) * lr
    return w


# Doing inference to test our model
def test(X, Y, w):
    total_examples = X.shape[0]
    correct_results = np.sum(classify(X, w) == Y)
    success_percent = correct_results * 100 / total_examples
    print(
        "\nSuccess: %d/%d (%.2f%%)" % (correct_results, total_examples, success_percent)
    )


# Import the dataset
data_path = Path(__file__).parent / "police.txt"
if not data_path.exists():
    raise FileNotFoundError(
        f"{data_path} not found. Put police.txt next to main.py or adjust the path."
    )


# Prepare data
x1, x2, x3, y = np.loadtxt(data_path, skiprows=1, unpack=True)
X = np.column_stack((np.ones(x1.size), x1, x2, x3))  # adding bias term
Y = y.reshape(-1, 1)
w = train(X, Y, iterations=10000, lr=0.001)

# Test it
test(X, Y, w)

# Printing the weights
print(w)

# Printing the weights and the examples
col_names = ["bias", "x1", "x2", "x3"]
print("\nWeights:")
for name, weight in zip(col_names, w.ravel()):
    print(f"  {name:>6}: {weight:.6f}")

print(f"\nFirst {X.shape[0]} examples (columns: {', '.join(col_names)} | Y):")
header = (
    "{:>3} ".format("i")
    + " ".join(f"{c:>8}" for c in col_names)
    + "   {:>3}".format("Y")
)
print(header)
for i in range(X.shape[0]):
    feats = " ".join(f"{X[i, j]:8.4f}" for j in range(X.shape[1]))
    label = int(Y[i, 0])
    print(f"{i:3d} {feats}   {label:3d}")

# The bias is 1 because we added a column of ones to X. We are not using it yet
