import numpy as np

# Code of the sigmoid function, to compress values between 0 and 1
def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def forward(X, w):
  weighted_sum = np.matmul(X, w)
  return sigmoid(weighted_sum)

def classify(X, w):
  return np.round(forward(X, w))

def mse_loss(X, Y, w):
  return np.average((forward(X, w) - Y) ** 2)

def loss(X, Y, w):
    y_hat = forward(X, w)
    first_term = Y * np.log(y_hat)
    second_term = (1 - Y) * np.log(1 - y_hat)
    return -np.average(first_term + second_term)

def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]