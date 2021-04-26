from torch.utils.data import DataLoader
import os
import random
import torch
import numpy as np

from preprocessing import gauss_convolve_instance


class MyBaseDataset(DataLoader):
    def __init__(self):
        self.train_x = np.load('./data/train_x.npy')
        self.train_y = np.load('./data/train_y.npy')


    def __len__(self):
        return self.train_x.shape[0]


class MyDataset(MyBaseDataset):
    def __getitem__(self, index):
        current_x = self.train_x[index, :, np.array([1,2,3,5])]
        current_x = gauss_convolve_instance(current_x, [0, 1, 2], 0.5)
        current_y = np.array([np.argmax(self.train_y[index, :, 1]) / 120.])

        return current_x, current_y


class MyDataset1(MyBaseDataset):
    def __getitem__(self, index):
        current_x = torch.from_numpy(self.train_x[index, :, np.array([1,2,3,5])])
        # current_x = torch.from_numpy(self.train_x[index, :, 1][np.newaxis, :])
        current_x[0] = (current_x[0] - 1) * 50
        if self.train_y[index, :, 1].max() > 1.005:
            # current_y = 1
            current_y = np.array([1])
        else:
            # current_y = 0
            current_y = np.array([0])

        return current_x, current_y