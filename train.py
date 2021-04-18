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


BATCH_SIZE = 16

def train():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    data_col_idx = 1 # 1 open, 2 high

    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    os.makedirs('./model', exist_ok=True)

    model = MyModel().to(device)
    epoch = 0

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    cross_criterion = nn.CrossEntropyLoss()

    while epoch <= 500:
        epoch += 1

        model.train()

        pbar = tqdm(range(len(dataloader)))
        pbar.set_description('Epoch {}'.format(epoch))
        total_loss = .0
        correct = 0

        for idx, (train_x, train_y) in enumerate(dataloader):
            train_x = train_x.to(device, dtype=torch.float32)
            train_y = train_y.to(device, dtype=torch.long)

            output = model(train_x)

            loss = cross_criterion(output, train_y)
            correct += int((torch.argmax(output, 1) == train_y).float().sum())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().numpy()
            pbar.set_postfix_str('loss: {}, acc: {}'.format(np.around(total_loss / (idx + 1), 4), np.around(correct / ((idx + 1) * BATCH_SIZE), 4)))
            pbar.update()


if __name__ == '__main__':
    train()