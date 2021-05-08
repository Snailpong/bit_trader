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


def make_npy():
    train_x = get_array('./data/train_x_df.csv')
    train_y = get_array('./data/train_y_df.csv')
    test_x = get_array('./data/test_x_df.csv')

    # train_x_array (7661, 1380, 10), train_y_array (7661, 120, 10), test_y_array (535, 120, 10)

    np.save('./data/train_x', train_x)
    np.save('./data/train_y', train_y)
    np.save('./data/test_x', test_x)


def make_only_increased():
    train_x = np.load('./data/train_x.npy')
    train_y = np.load('./data/train_y.npy')

    idx_increase = []

    for i in range(train_x.shape[0]):
        if train_y[i, :, 1].mean() >= 1.00:
            idx_increase.append(i)
        
    idx_increase = np.array(idx_increase)

    np.save('./data/train_x_increase', train_x[idx_increase])
    np.save('./data/train_y_increase', train_y[idx_increase])


def make_one_period(period, array):
    array_p = np.empty((array.shape[0], array.shape[1] // period, array.shape[2]))
    mod = array.shape[1] % period

    for i in range(array_p.shape[1]):
        start_time = mod + i * period
        array_p[:, i, 0] = array[:, start_time, 0]
        array_p[:, i, 1] = array[:, start_time, 1]
        array_p[:, i, 2] = np.max(array[:, start_time:start_time + period, 2], axis=1)
        array_p[:, i, 3] = np.min(array[:, start_time:start_time + period, 3], axis=1)
        array_p[:, i, 4] = array[:, start_time + period - 1, 4]
        array_p[:, i, 5:] = np.mean(array[:, start_time:start_time + period, 5:], axis=1)
        

    print(array[0, 0], array_p[0, 0])

    return array_p


def make_period(period):
    train_x = np.load('./data/train_x.npy')
    test_x = np.load('./data/test_x.npy')

    train_x_p = make_one_period(period, train_x)
    test_x_p = make_one_period(period, test_x)

    np.save('./data/train_x_' + str(period), train_x_p)
    np.save('./data/test_x_' + str(period), test_x_p)


if __name__ == '__main__':
    # make_npy()
    # make_only_increased()
    make_period(5)
    make_period(15)