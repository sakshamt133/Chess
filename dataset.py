import torch
import cv2 as cv
import numpy as np
import os
from torch.utils.data import Dataset


class Chess(Dataset):
    def __init__(self, path, transform=None):
        super(Chess, self).__init__()
        self.path = path
        self.folders = os.listdir(self.path)
        self.transform = transform
        self.length = (len(os.listdir(os.path.join(self.path, self.folders[0])))
                       + len(os.listdir(os.path.join(self.path, self.folders[1])))
                       + len(os.listdir(os.path.join(self.path, self.folders[2])))
                       + len(os.listdir(os.path.join(self.path, self.folders[3]))))

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        images = []
        labels = []
        for i, folder in enumerate(self.folders):
            new_path = os.path.join(self.path, folder)
            for img in os.listdir(new_path):
                images.append(os.path.join(new_path, img))
                labels.append(i)

        img = images[item]
        label = labels[item]
        img = cv.imread(img)
        img = cv.resize(img, (300, 300))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = np.asarray(img)
        img = self.transform(img)
        label = np.array(label)
        label = torch.from_numpy(label)
        return img, label
