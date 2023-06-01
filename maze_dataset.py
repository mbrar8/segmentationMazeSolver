import os
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class MazeDataset(Dataset):
    def __init__(self, transform=None):
        self.image_dir = "saved_imgs"
        self.mask_dir = "mask_imgs"
        self.transform = transform

        self.image_files = sorted(os.listdir(self.image_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.image_files[index])
        mask_path = os.path.join(self.mask_dir, self.mask_files[index])

        img = plt.imread(img_path)
        mask = plt.imread(mask_path)

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask

