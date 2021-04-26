import math
import numpy as np
import pandas as pd


def df2d_to_array3d(df_2d):
    feature_size = df_2d.iloc[:,2:].shape[1]
    time_size = len(df_2d.time.value_counts())
    sample_size = len(df_2d.sample_id.value_counts())
    sample_index = df_2d.sample_id.value_counts().index
    array_3d = df_2d.iloc[:,2:].values.reshape([sample_size, time_size, feature_size])
    
    return array_3d


def get_array(file_path):
    df = pd.read_csv(file_path)
    array = df2d_to_array3d(df)

    return array


def gauss_1d(sigma):
    length = math.ceil(((sigma * 6 + 1)) // 2) * 2 + 1
    a = np.array(range(length))
    a = a - (length // 2)

    gauss = lambda x: math.exp( -(x ** 2) / (2 * (sigma ** 2)))
    vfunc = np.vectorize(gauss)
    a = vfunc(a)

    a = a / np.sum(a)

    return a


def convolve_1d(array, filter):
    padding = math.ceil((filter.shape[0] - 1) / 2)
    npad = (padding, padding)
    a = np.zeros(array.shape[0])
    array = np.pad(array, npad, 'edge')
    
    for i in range(a.shape[0]):
        a[i] = np.sum(np.multiply(array[i:i+2*padding+1], filter))

    return a


def gauss_convolve_instance(array, rows, sigma):
    filter = gauss_1d(sigma)
    for row in rows:
        array[:, row] = convolve_1d(array[:, row], filter)

    return array


if __name__ == '__main__':
    train_x_array = get_array('./data/train_x_df.csv')
    train_y_array = get_array('./data/train_y_df.csv')
    test_x_array = get_array('./data/test_x_df.csv')

    # train_x_array (7661, 1380, 10), train_y_array (7661, 120, 10), test_y_array (535, 120, 10)

    np.save('./data/train_x', train_x_array)
    np.save('./data/train_y', train_y_array)
    np.save('./data/test_x', test_x_array)