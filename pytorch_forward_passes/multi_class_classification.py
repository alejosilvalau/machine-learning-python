import torch
import torch.nn as nn

input_data = torch.load('input_data.pt')
n_classes = 3

# Create multi-class classification model 
model = nn.Sequential(
    nn.Linear(6, 4),   # First linear layer: 6 input features → 4 hidden units
    nn.Linear(4, n_classes),   # Second linear layer: 4 hidden units → 1 output unit
    nn.Softmax(dim=-1)       # Softmax activation function for multi-class classification
)

output = model(input_data)
print("Output probabilities:")
print(output)