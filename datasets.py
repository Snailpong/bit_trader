from torch.utils.data import DataLoader
import os
import random
import torch
import numpy as np

from preprocessing import gauss_convolve_instance, gauss_1d, convolve_1d


def get_dataset(file_x, file_y):
    x = np.load(file_x)
    y = np.load(file_y)

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
        self.filter = gauss_1d(0.5)


    def __len__(self):
        return self.x.shape[0]


    def smoothing_x(self, index):
        current_x = self.x[index, :, np.array([1,2,3,5])]
        current_x[:, 5] /= np.mean(current_x[:, 5])
        current_x = gauss_convolve_instance(current_x, [0, 1, 2], 0.5)

        return current_x


class MyDataset(MyBaseDataset):
    def __getitem__(self, index):
        current_x = super().smoothing_x(index)
        current_y = np.array([np.argmax(convolve_1d(self.y[index, :, 1], self.filter)) / 120.])

        return current_x, current_y


class MyDataset1(MyBaseDataset):
    def __getitem__(self, index):
        current_x = super().smoothing_x(index)
        blur_y = convolve_1d(self.y[index, :, 1], self.filter)
        current_y = np.array([np.mean(blur_y)])

        # if blur_y.mean() > 1.00:
            # current_y = 1
            # current_y = np.array([1])
        # else:
            # current_y = 0
            # current_y = np.array([0])

        return current_x, current_y