import numpy as np
import pandas as pd
import gc
import math
import os
import random
import matplotlib.pyplot as plt
import datetime

import numpy as np

from datasets import MyDataset, get_dataset
from utils import init_device_seed, write_val_csv
from preprocessing import gauss_1d, convolve_1d


train, val, indicate = get_dataset('./data/train_x_5.npy', './data/train_y.npy')

train_chart = train[0][:, :, 1]
val_chart = val[0][:, :, 1]

filter = gauss_1d(0.5)

for i in range(train_chart.shape[0]):
    # train_chart[i] -= 1
    train_chart[i] = convolve_1d(train_chart[i], filter)
    train_chart[i] -= train_chart[i].min()
    train_chart[i] /= train_chart[i].max()

tp, tn, fp, fn = 0, 0, 0, 0

for i in range(val_chart.shape[0]):
    val_chart[i] -= 1
    val_chart[i] = convolve_1d(val_chart[i], filter)
    val_chart[i] -= val_chart[i].min()
    val_chart[i] /= val_chart[i].max()
    max_similarity = -1
    max_label = -1

    for j in range(train_chart.shape[0]):
        similarity = np.dot(val_chart[i], train_chart[j]) / np.linalg.norm(train_chart[j]) / np.linalg.norm(val_chart[i])

        if max_similarity < similarity:
            max_similarity = similarity
            max_label = j

    max_day = np.argmax(train[1][i, :, 1])
    train_mean = np.mean(train[1][i, :, 1]).mean()
    val_mean = np.mean(val[1][i, :, 1]).mean()
    train_max_price = train[1][i, max_day, 1]
    val_max_price = val[1][i, max_day, 1]
    # print(similarity, train_mean, val_mean, train_max_price, val_max_price)

    if train_mean > 1.0 and val_mean > 1.0: tp += 1
    if train_mean < 1.0 and val_mean < 1.0: tn += 1
    if train_mean < 1.0 and val_mean > 1.0: fp += 1
    if train_mean > 1.0 and val_mean < 1.0: fn += 1

    print(i, tp, tn, fp, fn, end='\r')


