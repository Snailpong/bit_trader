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
from models import MyModel2
from utils import init_device_seed, write_val_csv


BATCH_SIZE = 32


def train():
    device = init_device_seed()

    train, val, indicate = get_dataset('./data/train_x_15.npy', './data/train_y.npy')
    train_dataset = MyDataset1(train)
    val_dataset = MyDataset1(val)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    os.makedirs('./model', exist_ok=True)

    model = MyModel2(1, 1024).to(device)
    epoch = 0
    min_total_val_loss = 9999

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    cross_criterion = nn.CrossEntropyLoss()
    mae_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()

    while epoch <= 50:
        epoch += 1

        model.train()

        total_loss = .0
        correct = 0

        for idx, (train_x, train_y) in enumerate(train_dataloader):
            train_x = train_x.to(device, dtype=torch.float32)
            train_y = train_y.to(device, dtype=torch.float32)
            output = model(train_x)

            loss = mse_criterion(output, train_y)
            correct += int((torch.argmax(output, 1) == train_y).float().sum())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().numpy()
            print('\rEpoch {}: {}/{}, loss: {}'.format(epoch, idx, len(train_dataloader), np.around(total_loss / (idx + 1), 4)), end='')

        total_val_loss = .0
        tp, tn, fp, fn = 0, 0, 0, 0
        labels = []

        with torch.no_grad():
            for idx, (val_x, val_y) in enumerate(val_dataloader):
                val_x = val_x.to(device, dtype=torch.float32)
                val_y = val_y.to(device, dtype=torch.float32)

                output = model(val_x)
                # label = torch.argmax(output, 1)
                labels.append(output.detach().cpu().numpy())

                loss = mse_criterion(output, val_y)
                total_val_loss += loss.detach().cpu().numpy()

            #     tp += int(torch.logical_and((label == val_y), label == 1).float().sum())
            #     tn += int(torch.logical_and((label == val_y), label == 0).float().sum())
            #     fp += int(torch.logical_and((label != val_y), label == 1).float().sum())
            #     fn += int(torch.logical_and((label != val_y), label == 0).float().sum())

            # print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp, tn, fp, fn))

        print('\tval loss: ' + str(total_val_loss / len(val_dataloader)))

        if min_total_val_loss > total_val_loss:
            min_total_val_loss = total_val_loss
            torch.save(model.state_dict(), './model/isup')
            write_val_csv('./result/validation_isup.csv', labels, indicate, False)


if __name__ == '__main__':
    train()