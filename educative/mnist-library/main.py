# A binary classifier that recognizes one of the digits in MNIST.

import numpy as np
import os

# Limit CPU threads to prevent system freeze
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"


# Applying Logistic Regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(X, w):
    weighted_sum = np.matmul(X, w)
    return sigmoid(weighted_sum)


def classify(X, w):
    return np.round(forward(X, w))


# Computing Loss over using logistic regression
def loss(X, Y, w):
    y_hat = forward(X, w)
    # Add numerical stability
    y_hat = np.clip(y_hat, 1e-8, 1 - 1e-8)
    first_term = Y * np.log(y_hat)
    second_term = (1 - Y) * np.log(1 - y_hat)
    return -np.average(first_term + second_term)


# calculating gradient
def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]


# Use minibatch training to reduce peak CPU/memory usage
def train(X, Y, iterations, lr, batch_size=2048):
    # Ensure float32 and proper shapes
    X = X.astype(np.float32)
    Y = Y.reshape(-1, 1).astype(np.float32)
    w = np.zeros((X.shape[1], 1), dtype=np.float32)
    n = X.shape[0]

    for i in range(iterations):
        # Minibatch SGD
        perm = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            Xb = X[idx]
            Yb = Y[idx]
            w -= gradient(Xb, Yb, w) * lr

        # Print progress less frequently
        if (i % 20) == 0:
            print("Iteration %4d => Loss: %.6f" % (i, loss(X, Y, w)))
    return w


# Doing inference to test our model
def test(X, Y, w, digit):
    X = X.astype(np.float32)
    Y = Y.reshape(-1, 1).astype(np.float32)
    total_examples = X.shape[0]
    correct_results = np.sum(classify(X, w) == Y)
    success_percent = correct_results * 100 / total_examples
    print(
        "Correct classifications for digit %d: %d/%d (%.2f%%)"
        % (digit, correct_results, total_examples, success_percent)
    )


# Test it after loading the data
import mnist as data

for digit in range(10):
    print(f"\n--- Training classifier for digit {digit} ---")
    w = train(
        data.X_train, data.Y_train[digit], iterations=100, lr=1e-3, batch_size=2048
    )
    test(data.X_test, data.Y_test[digit], w, digit)

"""
Loaded MNIST dataset from TensorFlow:
  X_train shape: (60000, 785)
  X_test shape: (10000, 785)
  Y_train: 10 binary classifiers, each shape (60000, 1)
  Y_test: 10 binary classifiers, each shape (10000, 1)
  Dataset info: {'train': <SplitInfo num_examples=60000, num_shards=1>, 'test': <SplitInfo num_examples=10000, num_shards=1>}
Correct classifications for digit 0: 9020/10000 (90.20%)
Correct classifications for digit 1: 8865/10000 (88.65%)
Correct classifications for digit 2: 8968/10000 (89.68%)
Correct classifications for digit 3: 8990/10000 (89.90%)
Correct classifications for digit 4: 9018/10000 (90.18%)
Correct classifications for digit 5: 9108/10000 (91.08%)
Correct classifications for digit 6: 9042/10000 (90.42%)
Correct classifications for digit 7: 8972/10000 (89.72%)
Correct classifications for digit 8: 9026/10000 (90.26%)
Correct classifications for digit 9: 8991/10000 (89.91%)
"""
