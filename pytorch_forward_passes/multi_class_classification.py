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

# Output probabilities:
# tensor([[0.1784, 0.5115, 0.3101],
#         [0.3676, 0.3856, 0.2469],
#         [0.3183, 0.2689, 0.4128],
#         [0.2957, 0.4026, 0.3017],
#         [0.3050, 0.3820, 0.3130]], grad_fn=<SoftmaxBackward0>)
#
# This means that rows 1, 2, 4 and 5 are one animal type. And the row 3 is another animal type.