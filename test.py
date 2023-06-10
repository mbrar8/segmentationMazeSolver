import torch
import torch.nn as nn
from maze_generator import generate_maze
from termcolor import colored
from unet import UNet
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

model = UNet()
model.load_state_dict(torch.load('unet_maze100.pth'))
model.eval()
size = 100
# Labels were converted from 0/1 to 0/0.0039 so adjusted threshold to match
# Can figure out why and fix later
threshold = 0.002
num_mazes = 10

for n in range(num_mazes):
    maze, mask = generate_maze(1, size, 'BFS', False)


    trm = transforms.Compose([transforms.ToTensor()])

    tensor_maze = trm(maze)

    tensor_maze = tensor_maze[None, :]


    output = torch.sigmoid(model(tensor_maze))


    if size > 30:
        # Save image instead of printing if too large (won't fit on terminal screen)
        output_img = np.zeros((size,size,3), dtype=np.uint8)
        true_img = np.zeros((size,size,3), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                if maze[i, j, 0] == 255:
                    true_img[i,j] = [255,0,0]
                    if output[0,0,i,j] < threshold:
                        output_img[i,j] = [255,0,0]
                    else:
                        output_img[i,j] = [0,255,0]
                elif maze[i,j,2] == 255:
                    if mask[i,j] == 1:
                        true_img[i,j] = [0,255,0]
                    else:
                        true_img[i,j] = [0,0,255]
                    if output[0,0,i,j] < threshold:
                        output_img[i,j] = [0,0,255]
                    else:
                        output_img[i,j] = [0,255,0]

        plt.imshow(true_img)
        plt.axis('off')
        plt.savefig("test_true_" + str(n) + ".png")
        plt.show()

        plt.imshow(output_img)
        plt.axis('off')
        plt.savefig("test_output_" + str(n) + ".png")
        plt.show()

        continue

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

