import numpy as np
import pandas as pd
import gc
import torch
import random
import math
import os
import matplotlib.pyplot as plt

from tqdm import tqdm

from models import MyModel
from preprocessing import gauss_convolve_instance
from utils import init_device_seed


def test():
    device = init_device_seed()

    data_path = './data'
    test_dataset = np.load('./data/test_x_5.npy')

    os.makedirs('./result', exist_ok=True)

    model = MyModel(1, 3584).to(device)
    model.load_state_dict(torch.load('./model/maxday', map_location=device))
    model.eval()

    f = open('./result/submission.csv', 'w')
    f.write('sample_id,buy_quantity,sell_time\n')

    for index in range(535):
        print('\r{}/535'.format(index), end=' ')

        test_x = test_dataset[index, :, np.array([1, 2, 3, 5])]
        test_x[:, 5] /= np.mean(test_x[:, 5])
        test_x = gauss_convolve_instance(test_x, [0, 1, 2], 0.5)
        test_x = torch.from_numpy(test_x)
        test_x = torch.unsqueeze(test_x, 0).to(device, dtype=torch.float32)

        output = model(test_x)
        label = int(np.clip(np.around(output.detach().cpu().numpy()[0, 0] * 120.), 0, 119))

        f.write('{},1.0,{}\n'.format(index + 7661, label))

    f.close()


if __name__ == '__main__':
    test()