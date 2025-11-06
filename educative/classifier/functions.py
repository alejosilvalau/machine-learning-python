import numpy as np

# Code of the sigmoid function, to compress values between 0 and 1
def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def forward(X, w):
  weighted_sum = np.matmul(X, w)
  return sigmoid(weighted_sum)