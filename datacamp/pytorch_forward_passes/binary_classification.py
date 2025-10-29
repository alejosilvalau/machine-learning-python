import torch
import torch.nn as nn

input_data = torch.load('input_data.pt')

# Create binary classification model
model = nn.Sequential(
    nn.Linear(6, 4),   # First linear layer: 6 input features → 4 hidden units
    nn.Linear(4, 1),   # Second linear layer: 4 hidden units → 1 output unit
    nn.Sigmoid()       # Sigmoid activation function for binary classification
)

output = model(input_data)
print("Output probabilities:")
print(output)