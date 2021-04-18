import numpy as np
import matplotlib.pyplot as plt

from preprocessing import df2d_to_array3d, get_dataframe


def plot_series(x_series, y_series):

    plt.plot(x_series, label = 'input_series')
    plt.plot(np.arange(len(x_series), len(x_series)+len(y_series)),
             y_series, label = 'output_series')
    plt.axhline(1, c = 'red')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    train_x_df, train_y_df = get_dataframe()
    train_x_array = df2d_to_array3d(train_x_df)
    train_y_array = df2d_to_array3d(train_y_df)

    idx = 2100

    plot_series(train_x_array[idx, :, 1], train_y_array[idx, :, 1])

