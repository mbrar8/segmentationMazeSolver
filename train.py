import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from unet import UNet
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import os
from maze_dataset import MazeDataset
from maze_generator_dataset import MazeGeneratorDataset
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument("--lr", default=0.001)
parser.add_argument("--batch", default=16)
parser.add_argument("--epoch", default=50)
parser.add_argument("--data", choices=[0, 1], 
        help="Maze Dataset. 0: MazeDataset, 1: MazeGeneratorDataset", default=0)
parser.add_argument("--data_len", 
        help="Length for MazeGeneratorDataset", default=1000)

args = parser.parse_args()


lr = args.lr
batch_size = args.batch
epochs = args.epoch

model = UNet()

lossfxn = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=lr)

maze_dataset = None

if args.data == 0:
    maze_dataset = MazeDataset(transform=ToTensor())
elif args.data == 1:
    maze_dataset = MazeGeneratorDataset(args.data_len, transform=ToTensor())


train_dataset, val_dataset = random_split(maze_dataset, [0.8, 0.2])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


losses = []
min_val_loss = np.inf
early_stopping = 0
early_stopping_limit = 5
print("Starting")
for epoch in range(epochs):
    tl = 0.0
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)

        loss = lossfxn(torch.sigmoid(output), labels)

        loss.backward()
        optimizer.step()

        tl += loss.item()

    currloss = tl / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {currloss:.4f}")

    vl = 0.0
    model.eval()
    for images, labels in val_loader:
        output = model(images)
        loss = lossfxn(torch.sigmoid(output), labels)
        vl += loss.item()

    vlloss = vl / len(val_loader)
    losses.append((currloss, vlloss))
    print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {vlloss:.4f}")
    if min_val_loss > vlloss:
        min_val_loss = vlloss
        early_stopping = 0
    else:
        early_stopping += 1
        if early_stopping >= early_stopping_limit:
            print(f"Early Stopping at Epoch {epoch+1}, Min VLoss: {min_val_loss:.4f}")
            break

torch.save(model.state_dict(), 'unet_maze.pth')


with open('losses.txt', 'w') as fil:
    for loss in losses:
        fil.write(f"{loss}\n")

fil.close()
