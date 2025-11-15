import numpy as np


# Turns a matrix of labels * class into a one-hot encoded matrix
# where each label is represented as a vector of 0s and 1s
# with 1 at the index of the label.
def one_hot_encode(Y):
    n_labels = Y.shape[0]
    n_classes = 10
    encoded_Y = np.zeros((n_labels, n_classes))
    for i in range(n_labels):
        label = Y[i]
        encoded_Y[i][label] = 1
    return encoded_Y


# 60K labels, each a single digit from 0 to 9
Y_train_unencoded = load_labels(
    "/programming-machine-learning/data/mnist/train-labels-idx1-ubyte.gz"
)  # 60K labels, each consisting of 10 one-hot encoded elements
Y_train = one_hot_encode(Y_train_unencoded)
# 10000 labels, each a single digit from 0 to 9
Y_test = load_labels(
    "/programming-machine-learning/data/mnist/t10k-labels-idx1-ubyte.gz"
)
