from torch.utils.data import DataLoader
import os
import random
import torch
import numpy as np

from preprocessing import gauss_convolve_instance


def get_dataset():
    x = np.load('./data/train_x.npy')
    y = np.load('./data/train_y.npy')

    total_len = x.shape[0]
    train_len = int(total_len * 0.8)
    indicate = np.arange(total_len)
    np.random.shuffle(indicate)

    x = x[indicate]
    y = y[indicate]

    train_x = x[:train_len]
    val_x = x[train_len:]
    train_y = y[:train_len]
    val_y = y[train_len:]

    return [train_x, train_y], [val_x, val_y], indicate


class MyBaseDataset(DataLoader):
    def __init__(self, dataset):
        self.x = dataset[0]
        self.y = dataset[1]


    def __len__(self):
        return self.x.shape[0]


class MyDataset(MyBaseDataset):
    def __getitem__(self, index):
        current_x = self.x[index, :, np.array([1,2,3,5])]
        current_x = gauss_convolve_instance(current_x, [0, 1, 2], 0.5)
        current_y = np.array([np.argmax(self.y[index, :, 1]) / 120.])

        return current_x, current_y


class MyDataset1(MyBaseDataset):
    def __getitem__(self, index):
        current_x = torch.from_numpy(self.x[index, :, np.array([1,2,3,5])])
        # current_x = torch.from_numpy(self.x[index, :, 1][np.newaxis, :])
        current_x[0] = (current_x[0] - 1) * 50
        if self.y[index, :, 1].max() > 1.005:
            # current_y = 1
            current_y = np.array([1])
        else:
            # current_y = 0
            current_y = np.array([0])

        return current_x, current_y