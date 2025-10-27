import torch
from torch.nn import CrossEntropyLoss

scores = torch.tensor([-5.2, 2.3, 0.5, 1.0])
one_hot_target = torch.tensor([0, 0, 1, 0])

criterion = CrossEntropyLoss()
print(criterion(scores.double(), one_hot_target.double()))
# Output: tensor(2.1635, dtype=torch.float64)

one_hot_target_2 = torch.tensor([1, 0, 0, 0])
print(criterion(scores.double(), one_hot_target_2.double()))
# Output: tensor(7.8635, dtype=torch.float64)