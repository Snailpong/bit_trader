import numpy as np
import pandas as pd
import torch
import gc
import math
import os
import random
import matplotlib.pyplot as plt
import datetime

from torch import nn, optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

from datasets import MyDataset1, get_dataset
from models import MyModel, MyModel2
from utils import init_device_seed, write_val_csv
from preprocessing import gauss_convolve_instance


BATCH_SIZE = 32


def train():
    device = init_device_seed()

    _, val_5, _ = get_dataset('./data/train_x_5.npy', './data/train_y.npy')
    _, val_15, _ = get_dataset('./data/train_x_15.npy', './data/train_y.npy')
    os.makedirs('./model', exist_ok=True)

    model_day = MyModel(1, 3584).to(device)
    model_day.load_state_dict(torch.load('./model/maxday', map_location=device))
    model_day.eval()

    model_isup = MyModel2(1, 1024).to(device)
    model_isup.load_state_dict(torch.load('./model/isup', map_location=device))
    model_isup.eval()

    money = 10000

    for index in range(val_5[0].shape[0]):
        print('\r{}/{}'.format(index, val_5[0].shape[0]), end=' ')

        val_5_x = val_5[0][index, :, np.array([1, 2, 3, 5])]
        val_5_x[:, 5] /= np.mean(val_5_x[:, 5])
        val_5_x = gauss_convolve_instance(val_5_x, [0, 1, 2], 0.5)
        val_5_x = torch.from_numpy(val_5_x)
        val_5_x = torch.unsqueeze(val_5_x, 0).to(device, dtype=torch.float32)

        output_day = model_day(val_5_x)
        label = int(np.clip(np.around(output_day.detach().cpu().numpy()[0, 0] * 120.), 0, 119))

        val_15_x = val_15[0][index, :, np.array([1, 2, 3, 5])]
        val_15_x[:, 5] /= np.mean(val_15_x[:, 5])
        val_15_x = gauss_convolve_instance(val_15_x, [0, 1, 2], 0.5)
        val_15_x = torch.from_numpy(val_15_x)
        val_15_x = torch.unsqueeze(val_15_x, 0).to(device, dtype=torch.float32)

        output_day = model_isup(val_15_x)
        isup = output_day.detach().cpu().numpy()[0, 0]

        rate = val_5[1][index, label, 1] * 0.9995 * 0.9995

        # if np.mean(val_5[0][index, :, 1]) < 1:
        if isup >= 1:
            money *= rate


        print(rate, money, end='')



if __name__ == '__main__':
    train()