import dataset  # Import your dataset module
import numpy as np
import matplotlib.pyplot as plt

DIGIT = 5

# Use the data from dataset.py
X = dataset.X
Y = dataset.Y

digits = X[Y == DIGIT]
np.random.shuffle(digits)

rows, columns = 3, 15
fig = plt.figure()
for i in range(rows * columns):
    ax = fig.add_subplot(rows, columns, i + 1)
    ax.axis("off")
    ax.imshow(digits[i][:-1].reshape((28, 28)), cmap="Greys")
plt.savefig("./educative/MNIST Library/mnist_digits.png")
