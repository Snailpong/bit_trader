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


train, val, indicate = get_dataset('./data/train_x_15.npy', './data/train_y.npy')

train_chart = train[0][:, :, 1]
val_chart = val[0][:, :, 1]

for i in range(train_chart.shape[0]):
    train_chart[i] -= train_chart[i].min()
    train_chart[i] /= train_chart[i].max()

for i in range(val_chart.shape[0]):
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
    print(similarity, np.mean(train[1][i, :, 1]).mean(), np.mean(val[1][i, :, 1]).mean(), train[1][i, max_day, 1], val[1][i, max_day, 1])

