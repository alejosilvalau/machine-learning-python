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
