import pandas as pd
import numpy as np
from tinygrad.tensor import Tensor
import torch
import torch.optim as optim
import torch.nn.functional as F
import tensorflow as tf
from matplotlib import pyplot as plt

# load and preprocess dataset
df = pd.read_csv('daily-website-visitors.csv')
# strip commas and convert to int
df['Unique.Visits'] = df['Unique.Visits'].str.replace(',', '').astype(int)
y = df['Unique.Visits'].values

# preprocess
y = y.reshape(-1, 1)
split = int(len(y) * 0.8)
train, test = y[:split], y[split:]
scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
train_scaled, test_scaled = scaler(train), scaler(test)

# tinygrad model setup
class TinyModel:
    def __init__(self): self.l1 = Tensor.randn(1, 64)
    def forward(self, x): return x.dot(self.l1).relu().dot(Tensor.randn(64, 1))

# pytorch model setup
class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(1, 64)
        self.relu = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(64, 1)
    def forward(self, x): return self.l2(self.relu(self.l1(x)))


# convert numpy arrays to torch tensors
train_scaled_torch = torch.Tensor(train_scaled).view(-1, 1)
test_scaled_torch = torch.Tensor(test_scaled).view(-1, 1)

# instantiate the model, loss function, and optimizer
model = TorchModel()
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# training loop
epochs = 1000
train_losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(train_scaled_torch)
    loss = criterion(output, train_scaled_torch)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    if epoch % 100 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# plot the training loss
plt.plot(train_losses, label='Training loss')
plt.legend()
plt.show()

# prediction vs actual
with torch.no_grad():
    predicted = model(train_scaled_torch).view(-1).numpy()
    actual = train_scaled_torch.view(-1).numpy()
    plt.figure(figsize=(10,5))
    plt.plot(actual, label='Actual Data')
    plt.plot(predicted, label='Predicted Data')
    plt.legend()
    plt.show()