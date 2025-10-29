# The goal is to train a model to predict the type of animal based on its features.
#
# The types categories are: 0 for Bird, 1 for Mammal, and 2 for Reptile.
#
# The model should be able to learn from the features and
# predict the correct category given the features of an animal.

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch

animals = pd.read_csv('animal_dataset.csv')

features = animals.iloc[:, 1: -1]

x = features.to_numpy()
print("Features:", x)

'''
Features:
[[0 1 1 0 0 2 1]
 [0 1 1 0 1 2 1]
 [1 0 0 1 1 4 1]
 [1 0 0 1 0 4 1]
 [0 0 1 0 1 4 1]]
'''

target = animals.iloc[:, -1]
y = target.to_numpy()
print("Target:", y)

# Target: [0 0 1 1 2]

dataset = TensorDataset(torch.tensor(x), torch.tensor(y))
input_sample, label_sample = dataset[0]
print("Input sample:", input_sample)
print("Label sample:", label_sample)

# Input sample: tensor([0, 1, 1, 0, 0, 2, 1])
# Label sample: tensor(0)

batch_size = 2
shuffle = True

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
for batch in dataloader:
    inputs, labels = batch
    print("Batch inputs:", inputs)
    print("Batch labels:", labels)

'''
Batch inputs: 
tensor([[1, 0, 0, 1, 1, 4, 1],
        [0, 1, 1, 0, 1, 2, 1]])
Batch labels: tensor([1, 0])

Batch inputs: 
tensor([[0, 0, 1, 0, 1, 4, 1],
        [1, 0, 0, 1, 0, 4, 1]])
Batch labels: tensor([2, 1])

Batch inputs: 
tensor([[0, 1, 1, 0, 0, 2, 1]])
Batch labels: tensor([0])
'''
