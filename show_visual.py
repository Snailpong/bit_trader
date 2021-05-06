import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_series(x_series, y_series):

    plt.plot(x_series, label = 'input_series')
    plt.plot(np.arange(len(x_series), len(x_series)+len(y_series)),
             y_series, label = 'output_series')
    plt.axhline(1, c = 'red')
    plt.legend()


def show_example():
    x = np.load('./data/train_x.npy')
    y = np.load('./data/train_y.npy')

    idx = 2100

    plot_series(x[idx, :, 1], y[idx, :, 1])
    plt.show()


def show_trained_result():
    x = np.load('./data/train_x_increase.npy')
    y = np.load('./data/train_y_increase.npy')

    val = pd.read_csv('./result/validation.csv')
    len_val = len(val)

    for i in range(6):
        plt.subplot(2, 3, i+1)
        idx = random.randrange(len_val)

        plot_series(x[idx, :, 1], y[idx, :, 1])
        plt.axvline(1380 + val.loc[idx, 'sell_time'], c = 'blue')
        plt.axhline(y[idx, val.loc[idx, 'sell_time'], 1], c = 'green')
    plt.show()


if __name__ == '__main__':
    show_trained_result()