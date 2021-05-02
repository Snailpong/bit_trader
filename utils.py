import os
import torch
import random
import numpy as np


def init_device_seed():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    return device


def write_val_csv(labels, indicate):
    index = [0, 0]
    val_indicate = indicate[int(indicate.shape[0] * 0.8):]

    f = open('./result/validation.csv', 'w')
    f.write('sample_id,buy_quantity,sell_time\n')

    for i in range(val_indicate.shape[0]):
        label = int(np.clip(np.around(labels[index[0]][index[1], 0] * 120.), 0, 119))
        f.write('{},1.0,{}\n'.format(val_indicate[i], label))

        index[1] += 1
        if index[1] == labels[index[0]].shape[0]:
            index[0] += 1
            index[1] = 0

    f.close()