import torch
import torch.nn as nn



# Create input data of shape 5x6 (5 animals, 6 features)
input_data = torch.tensor([
    [-0.4421,  1.5207,  2.0607, -0.3647,  0.4691,  0.0946],
    [-0.9155, -0.0475, -1.3645,  0.6336, -1.9520, -0.3398],
    [ 0.7406,  1.6763, -0.8511,  0.2432,  0.1123, -0.0633],
    [-1.6630, -0.0718, -0.1285,  0.5396, -0.0288, -0.8622],
    [-0.7413,  1.7920, -0.0883, -0.6685,  0.4745, -0.4245]
])

torch.save(input_data, 'input_data.pt')