import torch
import torch.nn as nn

input_data = torch.load('input_data.pt')

model = nn.Sequential(
    nn.Linear(6, 4),
    nn.Linear(4, 1)
)
output = model(input_data)
print("Output values:")
print(output)

# Output values:
# tensor([[ 0.9853],
#         [-0.2355],
#         [ 0.7939],
#         [-0.0961],
#         [ 0.6047]], grad_fn=<AddmmBackward0>)
#
# This is the predicted value for each animal based on the 6 features.