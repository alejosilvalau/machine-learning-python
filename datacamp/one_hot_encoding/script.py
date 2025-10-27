import torch
import torch.nn.functional as F

print(F.one_hot(torch.tensor(0), num_classes = 4))
# Output: tensor([1, 0, 0, 0])

print(F.one_hot(torch.tensor(1), num_classes = 4))
# Output: tensor([0, 1, 0, 0])

print(F.one_hot(torch.tensor(2), num_classes = 4))
# Output: tensor([0, 0, 1, 0])

print(F.one_hot(torch.tensor(3), num_classes = 4))
# Output: tensor([0, 0, 0, 1])