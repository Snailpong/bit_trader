import torch
import torchvision.models as models
from torch import nn
from torch.nn import functional as F


class BaseModel(nn.Module):
    def __init__(self, fc):
        super(BaseModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(4, 64, 7, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(5, 2),
            nn.Conv1d(64, 128, 5, 1),
            nn.Conv1d(128, 128, 5, 1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(5, 2),
            nn.Conv1d(128, 256, 5, 1),
            nn.Conv1d(256, 256, 5, 1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(5, 2),
            nn.Conv1d(256, 512, 5, 1),
            nn.Conv1d(512, 512, 5, 1),
            nn.LeakyReLU(),
            nn.MaxPool1d(5, 2),
            nn.Flatten(),
            nn.Linear(fc, 64),
            nn.Linear(64, 16)
        )


    def forward(self, x):
        x = self.layers(x)
        return x


class MyModel(nn.Module):
    def __init__(self, channel, fc):
        super(MyModel, self).__init__()
        self.base = BaseModel(fc)
        self.final = nn.Linear(16, channel)

    
    def forward(self, x):
        x = self.base(x)
        x = self.final(x)
        return x


class MyModel2(nn.Module):
    def __init__(self, channel, fc):
        super(MyModel2, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(4, 128, 7, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.MaxPool1d(5, 2),
            nn.Conv1d(128, 256, 5, 1),
            nn.Conv1d(256, 256, 5, 1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(5, 2),
            nn.Conv1d(256, 512, 5, 1),
            nn.Conv1d(512, 512, 5, 1),
            nn.LeakyReLU(),
            nn.MaxPool1d(5, 2),
            nn.Flatten(),
            nn.Linear(fc, 64),
            nn.Linear(64, 16),
            nn.Linear(16, channel)
        )


    def forward(self, x):
        x = self.layers(x)
        return x


class MyModelLSTM(nn.Module):
    def __init__(self, channel, fc):
        super(MyModelLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=channel, hidden_size=16, num_layers=4, batch_first=True)
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )


    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.layers(x)
        return x