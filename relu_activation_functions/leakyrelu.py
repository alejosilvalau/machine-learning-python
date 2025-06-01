from torch import nn
import torch

leaky_relu_pytorch = nn.LeakyReLU(negative_slope = 0.05)

x = torch.tensor(-2.0)

output = leaky_relu_pytorch(x)
print(output)

'''
Output:
tensor(-0.1000)
'''