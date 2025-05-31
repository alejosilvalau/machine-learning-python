from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
import torch
import numpy as np
import pandas as pd

# Load data
data_science_salary = pd.read_csv('data_science_salary_dataset.csv')

# Extract features and target
features = data_science_salary.iloc[:, 1: -1]
target = data_science_salary.iloc[:, -1]

# Convert DataFrame to numpy arrays first, then to PyTorch tensors
features_tensor = torch.tensor(features.values).float()
target_tensor = torch.tensor(target.values).float()

# Check dimensions
print(f"Features shape: {features_tensor.shape}")
num_features = features_tensor.shape[1]
print(f"Number of features: {num_features}")

# Create the dataset and the dataloader
dataset = TensorDataset(features_tensor, target_tensor)

dataloader = DataLoader(
  dataset,
  batch_size=4,
  shuffle=True
)

# Create the model with correct input dimensions
model = nn.Sequential(
  nn.Linear(num_features, 2),
  nn.Linear(2, 1)
)

# Create the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

num_epochs = 1000

# Training loop
for epoch in range(num_epochs):
  loss = None  
  for data in dataloader:
    # Zero the gradients
    optimizer.zero_grad()

    # Get features and labels for this batch
    features_batch, labels_batch = data

    # Forward pass
    predictions = model(features_batch)

    # Calculate the loss and gradients
    loss = criterion(predictions, labels_batch.view(-1, 1)) 
    loss.backward()
    
    # Update the model parameters
    optimizer.step()
    
  # Print progress every 100 epochs
  if (epoch + 1) % 100 == 0 and loss is not None:
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

print("Training complete.")
# Print the final model parameters
print("Model parameters after training:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.data.numpy()}")


''' Output:
Features shape: torch.Size([5, 3])
Number of features: 3
Epoch 100/1000, Loss: 0.0010
Epoch 200/1000, Loss: 0.0224
Epoch 300/1000, Loss: 0.0056
Epoch 400/1000, Loss: 0.0004
Epoch 500/1000, Loss: 0.0098
Epoch 600/1000, Loss: 0.0106
Epoch 700/1000, Loss: 0.0051
Epoch 800/1000, Loss: 0.0064
Epoch 900/1000, Loss: 0.0092
Epoch 1000/1000, Loss: 0.0038
Training complete.
Model parameters after training:
0.weight: 
[[ 0.07043713  0.3430179  -0.20481107]
 [-0.35406384 -0.45947242 -0.50743616]]
0.bias: [-0.07867516 -0.16940738]
1.weight: [[-0.38825056  0.19231708]]
1.bias: [0.3155515]
(.venv) 
'''