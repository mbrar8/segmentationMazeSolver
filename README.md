# segmentationMazeSolver
Can segmentation models solve a maze? Just for fun

Training a UNet model to solve simple mazes.

maze_generator.py: Code for generating mazes using randomized Prim's algorithm. Mazes always start at bottom left and end at top right. Can run this file directly to save mazes as matplotlib images. 
Will result in higher resolution mazes for the same size

maze_dataset.py: Dataset class for mazes

maze_generator_dataset.py: Dataset class for maze that generates the maze during training without saving and opening an image version. Better suited for smaller mazes where maze generation doesn't take too long.

unet.py: UNet model

train.py: Train the model. --data flag specifies whether to use pre-generated saved mazes ("saved") or generate during training ("generate")

test.py: Test a saved model.

NOTE: Haven't added CLI arguments for most of these files yet, so parameters and file locations are mostly assumed

mask_imgs: Saved labels from generating and saving mazes. 
saved_imgs: Saved maze images from generating and saving mazes.

OutputResult/TrueResult: Example output from trained model.
losses.txt: Losses (avg train, avg val) for trained model.


Results: 
Model solves mazes quite well. Mostly seems to get the correct path, sometimes doesn't highlight some nodes on the path. Occasionally gets part of the path wrong, but beginning and end are mostly correct.
Makes sense as beginning and end are always the same positions. 


Future: 
Test on much more complex mazes (100x100) 20x20 results in very similar looking mazes most of the time--path that goes up and to the left. From a few test runs, the model seems to get paths that meander a little more
such as going back down, wrong. 
