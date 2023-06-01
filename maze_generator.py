import matplotlib.pyplot as plt
import numpy as np
from termcolor import colored
# Maze generator following randomized prim's algorithm


def checkSurround(maze, wall):
    total = 0
    if (wall[0] > 0 and maze[wall[0] - 1][wall[1]] == 2):
        total += 1
    if (wall[0] < size - 1 and maze[wall[0] + 1][wall[1]] == 2):
        total += 1
    if (wall[1] > 0 and maze[wall[0]][wall[1] - 1] == 2):
        total += 1
    if (wall[1] < size - 1 and maze[wall[0]][wall[1] + 1] == 2):
        total += 1

    if total < 2:
        return True
    return False

def solveMaze(maze):
    # Implements DFS
    parent = {}
    visited = []
    stack = []
    start = (0, 18)
    end = (19, 1)
    visited.append(start)
    stack.insert(0, start)
    while len(stack) > 0:
        pos = stack.pop()
        if pos == end:
            return parent
        if pos[0] > 0 and maze[pos[0] - 1][pos[1]] == 2 and (pos[0] - 1, pos[1]) not in visited:
            newPos = (pos[0] - 1, pos[1])
            visited.append(newPos)
            stack.insert(0,newPos)
            parent[newPos] = pos
        if pos[0] < size - 1 and maze[pos[0] + 1][pos[1]] == 2 and (pos[0] + 1, pos[1]) not in visited:
            newPos = (pos[0] + 1, pos[1])
            visited.append(newPos)
            stack.insert(0, newPos)
            parent[newPos] = pos
        if pos[1] > 0 and maze[pos[0]][pos[1] - 1] == 2 and (pos[0], pos[1] - 1) not in visited:
            newPos = (pos[0], pos[1] - 1)
            visited.append(newPos)
            stack.insert(0, newPos)
            parent[newPos] = pos
        if pos[1] < size - 1 and maze[pos[0]][pos[1] + 1] == 2 and (pos[0], pos[1] + 1) not in visited:
            newPos = (pos[0], pos[1] + 1)
            visited.append(newPos)
            stack.insert(0, newPos)
            parent[newPos] = pos





for maze_num in range(1000):
    size = 20
    # Create 1000 mazes - start is bottom left (19, 0) exit is top right (0, 19)
    maze = np.zeros((size, size))
    # 0 represents unvisited 1 represents a wall and 2 represents a space
    start = (np.random.randint(1, 20), np.random.randint(1, 20))
    maze[start[0]][start[1]] = 2
    wall_list = []
    if start[0] - 1 >= 0:
        wall_list.append((start[0] - 1, start[1]))
        maze[start[0] - 1][start[1]] = 1
    if start[0] + 1 < size:
        wall_list.append((start[0] + 1, start[1]))
        maze[start[0] + 1][start[1]] = 1
    if start[1] - 1 >= 0:
        wall_list.append((start[0], start[1] - 1))
        maze[start[0]][start[1] - 1] = 1
    if start[1] + 1 < size:
        wall_list.append((start[0], start[1] + 1))
        maze[start[0]][start[1] + 1] = 1

    k = 0
    while len(wall_list) > 0:
        #print(str(k) + " " + str(len(wall_list)))
        #k += 1
        print(maze)
        wall_index = np.random.randint(0, len(wall_list))
        wall = wall_list[wall_index]
        if wall[0] > 0 and wall[0] < size - 1 and maze[wall[0] - 1][wall[1]] == 0 and maze[wall[0] + 1][wall[1]] == 2:
            if checkSurround(maze, wall):
                maze[wall[0]][wall[1]] = 2
                
                if wall[1] > 0 and maze[wall[0]][wall[1] - 1] != 2:
                    maze[wall[0]][wall[1] - 1] = 1
                if wall[1] > 0 and (wall[0], wall[1] - 1) not in wall_list:
                    wall_list.append((wall[0], wall[1] - 1))

                if wall[1] < size - 1 and maze[wall[0]][wall[1] + 1] != 2:
                    maze[wall[0]][wall[1] + 1] = 1
                if wall[1] < size - 1 and (wall[0], wall[1] + 1) not in wall_list:
                    wall_list.append((wall[0], wall[1] + 1))

                if wall[0] > 0 and maze[wall[0] - 1][wall[1]] != 2:
                    maze[wall[0] - 1][wall[1]] = 1
                if wall[0] > 0 and (wall[0] - 1, wall[1]) not in wall_list:
                    wall_list.append((wall[0] - 1, wall[1]))
        
        elif wall[0] > 0 and wall[0] < size - 1 and maze[wall[0] + 1][wall[1]] == 0 and maze[wall[0] - 1][wall[1]] == 2:
            if checkSurround(maze, wall):
                maze[wall[0]][wall[1]] = 2
                
                if wall[1] > 0 and maze[wall[0]][wall[1] - 1] != 2:
                    maze[wall[0]][wall[1] - 1] = 1
                if wall[1] > 0 and (wall[0], wall[1] - 1) not in wall_list:
                    wall_list.append((wall[0], wall[1] - 1))

                if wall[1] < size - 1 and maze[wall[0]][wall[1] + 1] != 2:
                    maze[wall[0]][wall[1] + 1] = 1
                if wall[1] < size - 1 and (wall[0], wall[1] + 1) not in wall_list:
                    wall_list.append((wall[0], wall[1] + 1))

                if wall[0] < size - 1 and maze[wall[0] + 1][wall[1]] != 2:
                    maze[wall[0] + 1][wall[1]] = 1
                if wall[0] < size - 1 and (wall[0] + 1, wall[1]) not in wall_list:
                    wall_list.append((wall[0] + 1, wall[1]))

        elif wall[1] > 0 and wall[1] < size - 1 and maze[wall[0]][wall[1] - 1] == 0 and maze[wall[0]][wall[1] + 1] == 2:
            if checkSurround(maze, wall):
                maze[wall[0]][wall[1]] = 2
        
                if wall[0] > 0 and maze[wall[0] - 1][wall[1]] != 2:
                    maze[wall[0] - 1][wall[1]] = 1
                if wall[0] > 0 and (wall[0] - 1, wall[1]) not in wall_list:
                    wall_list.append((wall[0] - 1, wall[1]))

                if wall[0] < size - 1 and maze[wall[0] + 1][wall[1]] != 2:
                    maze[wall[0] + 1][wall[1]] = 1
                if wall[0] < size - 1 and (wall[0] + 1, wall[1]) not in wall_list:
                    wall_list.append((wall[0] + 1, wall[1]))

                if wall[1] > 0 and maze[wall[0]][wall[1] - 1] != 2:
                    maze[wall[0]][wall[1] - 1] = 1
                if wall[1] > 0 and (wall[0], wall[1] - 1) not in wall_list:
                    wall_list.append((wall[0], wall[1] - 1))

        elif wall[1] > 0 and wall[1] < size - 1 and maze[wall[0]][wall[1] + 1] == 0 and maze[wall[0]][wall[1] - 1] == 2:
            if checkSurround(maze, wall):
                maze[wall[0]][wall[1]] = 2
                
                if wall[0] > 0 and maze[wall[0] - 1][wall[1]] != 2:
                    maze[wall[0] - 1][wall[1]] = 1
                if wall[0] > 0 and (wall[0] - 1, wall[1]) not in wall_list:
                    wall_list.append((wall[0] - 1, wall[1]))

                if wall[0] < size - 1 and maze[wall[0] + 1][wall[1]] != 2:
                    maze[wall[0] + 1][wall[1]] = 1
                if wall[0] < size - 1 and (wall[0] + 1, wall[1]) not in wall_list:
                    wall_list.append((wall[0] + 1, wall[1]))

                if wall[1] < size - 1 and maze[wall[0]][wall[1] + 1] != 2:
                    maze[wall[0]][wall[1] + 1] = 1
                if wall[1] < size - 1 and (wall[0], wall[1] + 1) not in wall_list:
                    wall_list.append((wall[0], wall[1] + 1))
            
        wall_list.remove(wall)


    maze[19][1] = 2
    maze[0][18] = 2
    entrance = maze[1][18]
    entrance = 18
    exit = maze[18][1]
    exit = 1
    while (maze[1][entrance] != 2):
        maze[1][entrance] = 2
        entrance -= 1

    while (maze[18][exit] != 2):
        maze[18][exit] = 2
        exit += 1
        
    #plt.imshow(maze, cmap='binary')
    #plt.show()
    parent = solveMaze(maze)
    path = []
    node = (19, 1)
    while (node != (0, 18)):
        path.append(node)
        node = parent[node]

    path.append((0, 18))

    maze_img = np.zeros((len(maze), len(maze[0]), 3), dtype=np.uint8)
    maze_mask = np.zeros((len(maze), len(maze[0])), dtype=np.uint8)
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if (i, j) in path:
                # Crucial: Saved image doesn't include solved path (has to figure that itself)
                maze_img[i,j] = [0,0,255]
                maze_mask[i,j] = 1
                print(colored("\"", "green"), end = " ")
            elif maze[i][j] == 2.0:
                maze_img[i,j] = [0,0,255]
                maze_mask[i,j] = 0
                print(colored("\"", "blue"), end = " ")
            else:
                maze_img[i,j] = [255,0,0]
                maze_mask[i,j] = 0
                print(colored("#", "red"), end = " ")

        print("\n")

    plt.imshow(maze_img)
    plt.axis('off')
    plt.savefig("saved_imgs/maze_" + str(maze_num) + ".png")
    plt.show()

    plt.imshow(maze_mask)
    plt.axis('off')
    plt.savefig("mask_imgs/maze_mask_" + str(maze_num) + ".png")
    plt.show()



            
            

        





