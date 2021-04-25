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