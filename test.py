import torch
import torch.nn as nn
from maze_generator import generate_maze
from termcolor import colored
from unet import UNet
import torchvision.transforms as transforms


model = UNet()
model.load_state_dict(torch.load('unet_maze.pth'))
model.eval()
size = 20
# Labels were converted from 0/1 to 0/0.0039 so adjusted threshold to match
# Can figure out why and fix later
threshold = 0.002
num_mazes = 10

for n in range(num_mazes):
    maze, mask = generate_maze(1, size, False)


    trm = transforms.Compose([transforms.ToTensor()])

    tensor_maze = trm(maze)

    tensor_maze = tensor_maze[None, :]


    output = torch.sigmoid(model(tensor_maze))

    print(output.shape)

    print(torch.unique(output))

    print("True Maze")

    for i in range(size):
        for j in range(size):
            if maze[i, j, 0] == 255:
                print(colored('#', "red"), end = " ")
            elif maze[i, j, 2] == 255:
                if mask[i, j] == 1:
                    print(colored('\"', "green"), end = " ")
                else:
                    print(colored('\"', "blue"), end = " ")
        print("\n")


    print("Output Maze")

    for i in range(size):
        for j in range(size):
            if maze[i, j, 0] == 255:
                if output[0, 0, i, j] < threshold:
                    print(colored('#', "red"), end = " ")
                else:
                    print(colored('#', "green"), end = " ")
            elif maze[i, j, 2] == 255:
                if output[0, 0, i, j] < threshold:
                    print(colored('\"', "blue"), end = " ")
                else:
                    print(colored('\"', "green"), end = " ")
        print("\n")

