import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sea

# make path relative to this script file so it works regardless of CWD
data_path = Path(__file__).parent / "pizza.txt"
if not data_path.exists():
    raise FileNotFoundError(
        f"{data_path} not found. Put pizza.txt next to pizzeria.py or adjust the path."
    )

sea.set_theme()
plt.axis([0, 50, 0, 50])  # scale axes (0 to 50)
plt.xticks(fontsize=14)  # set x axis ticks
plt.yticks(fontsize=14)  # set y axis ticks
plt.xlabel("Reservations", fontsize=14)  # set x axis label
plt.ylabel("Pizzas", fontsize=14)  # set y axis label
X, Y = np.loadtxt(data_path, skiprows=1, unpack=True)  # load data
plt.plot(X, Y, "bo")  # plot data
plt.show()  # display chart


def predict(X, w):
    return X * w
