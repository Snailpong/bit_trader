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

from preprocessing import df2d_to_array3d, get_dataframe
from datasets import MyDataset
from models import MyModel


BATCH_SIZE = 32

def train():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    dataset = MyDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    os.makedirs('./model', exist_ok=True)

    model = MyModel().to(device)
    epoch = 0

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    cross_criterion = nn.CrossEntropyLoss()
    mae_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()

    while epoch <= 500:
        epoch += 1

        model.train()

        pbar = tqdm(range(len(train_dataloader)))
        pbar.set_description('Epoch {}'.format(epoch))
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
            pbar.set_postfix_str('loss: {}, acc: {}'.format(np.around(total_loss / (idx + 1), 4), np.around(correct / ((idx + 1) * BATCH_SIZE), 4)))
            pbar.update()

        with torch.no_grad():
            correct_val = 0
            tp, tn, fp, fn = 0, 0, 0, 0
            for idx, (val_x, val_y) in enumerate(test_dataloader):
                val_x = val_x.to(device, dtype=torch.float32)
                val_y = val_y.to(device, dtype=torch.long)

                output = model(val_x)
                # label = torch.argmax(output, 1)
                label = torch.round(output)
                # print(label.shape, val_y.shape)
                correct_val += int((label == val_y).float().sum())
                tp += int(torch.logical_and((label == val_y), label == 1).float().sum())
                tn += int(torch.logical_and((label == val_y), label == 0).float().sum())
                fp += int(torch.logical_and((label != val_y), label == 1).float().sum())
                fn += int(torch.logical_and((label != val_y), label == 0).float().sum())

            print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp, tn, fp, fn))



if __name__ == '__main__':
    train()