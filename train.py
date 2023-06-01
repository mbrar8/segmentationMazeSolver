import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from unet import UNet
from torch.utils.data import DataLoader
import os
from maze_dataset import MazeDataset
import matplotlib.pyplot as plt

lr = 0.001
batch_size = 16
epochs = 20

model = UNet()

lossfxn = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=lr)


train_dataset = MazeDataset(tranform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


losses = []
for epoch in range(epochs):
    rl = 0.0
    for images, labels in train_loader:
        output = model(images)

        loss = lossfxn(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rl += loss.item()

    currloss = rl / len(train_loader)
    losses.append(currloss)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {currloss:.4f}")


torch.save(model.state_dict(), 'unet_maze.pth')


with open('losses.txt', 'w') as fil:
    for loss in losses:
        fil.write(f"{loss}\n")

fil.close()
