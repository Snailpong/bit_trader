import torch
import torchvision.models as models
from torch import nn
from torch.nn import functional as F


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(2, 64, 7, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, 5, 2),
            nn.Conv1d(128, 128, 5, 1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 5, 2),
            nn.Conv1d(256, 256, 5, 1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 512, 5, 2),
            nn.Conv1d(512, 512, 5, 1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(82944, 64),
            nn.Linear(64, 2)
        )


    def forward(self, x):
        x = self.layers(x)
        return x