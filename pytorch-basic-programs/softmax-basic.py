import torch
import torch.nn as nn

input_tensor = torch.tensor([[1.0, -6.0, 2.5, -0.3, 1.2, 0.8]])

# Create a softmax function and apply it on input_tensor
softmax = nn.Softmax()
probabilities = softmax(input_tensor)
print(probabilities)