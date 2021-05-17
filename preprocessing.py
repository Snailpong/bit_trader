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
    is_increase = np.empty(train_x.shape[0])

    idx_increase = []
    count = 0

    for i in range(train_x.shape[0]):
        if train_y[i, :, 1].mean() >= 1.00:
            idx_increase.append(i)
            is_increase[i] = count
            count += 1
        else:
            is_increase[i] = -1
        
    idx_increase = np.array(idx_increase)

    # np.save('./data/train_x_increase', train_x[idx_increase])
    # np.save('./data/train_y_increase', train_y[idx_increase])
    np.save('./data/is_increase', train_y[idx_increase])


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


def get_MA(df, cycle=5, select_column='close'):
    ma = df.groupby('sample_id')[select_column].rolling(window=cycle).mean()
    ma = ma.reset_index()[select_column]
    ma.name = f'MA_{cycle}min'
    
    return pd.concat([df, ma], axis=1)
    
def get_MACD(df, short=12, long=26, t=9, select_column='close'):    
    ma_12 = df.groupby('sample_id').close.ewm(span=12).mean()
    ma_26 = df.groupby('sample_id').close.ewm(span=26).mean()
    
    macd = ma_12 - ma_26
    macds = macd.ewm(span=9).mean()  #Signal
    macdo = macd - macds # Osillator
    
    macd = macd.reset_index().close
    macd.name = 'macd'
    macds = macds.reset_index().close
    macds.name = 'macds'
    macdo = macdo.reset_index().close
    macdo.name = 'macdo'
    
    df = df.assign(macd=macd, macds=macds, macdo=macdo).dropna()
    
    return df

# Simple Moving Average
def SMA(df, period=30, column='close'):
    return df[column].rolling(window=period).mean()

# Relative Strength Index
# 매수 Signal = Under 30 Point
# 매도 Signal = Up 70 Point

def get_RSI(df, period=14, select_column='close'):
    data = df.copy()
    
    ma = data.groupby('sample_id')[select_column].diff(1)

    up = ma.copy()
    down = ma.copy()

    up[up < 0] = 0
    down[down > 0] = 0
    data['up'] = up
    data['down'] = down

    # AD, AU
    AVG_Gain = SMA(data.copy(), 14, column='up')
    AVG_Loss = abs(SMA(data.copy(), 14, column='down'))

    RS = AVG_Gain.dropna() / AVG_Loss.dropna()
    RS = RS.fillna(1)
    #### fillna parameter 조정 필요
    
    RSI = 100.0 - (100 / (1.0 + RS))
    
    data['RSI'] = RSI
    
    return data
    
def make_feature(df, cycle=5, select_column='close'):
    column = ['sample_id', 'time', 'coin_index', 'open', 'high', 'low', 'close', 'volume']
    df = df[column]
    
    df = get_RSI(df)
    df = get_MA(df)
    df = get_MACD(df)
    
    return df


def npy_to_2d(array):
    array_2d = np.empty((array.shape[0] * array.shape[1], 12))
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array_2d[i * array.shape[1] + j, 0] = i
            array_2d[i * array.shape[1] + j, 1] = j
            array_2d[i * array.shape[1] + j, 2:] = array[i, j]
    return array_2d


def make_period_with_feature():
    train_x = np.load('./data/train_x_15.npy')
    test_x = np.load('./data/test_x_15.npy')

    train_x_2d = npy_to_2d(train_x)
    test_x_2d = npy_to_2d(test_x)

    row_indices = ['sample_id', 'time', 'coin_index', 'open', 'high', 'low', 'close', 'volume', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av']

    train_df = pd.DataFrame(train_x_2d, columns=row_indices)
    train_df = train_df.astype({'sample_id': 'int', 'time': 'int'})
    print(train_df.head(5))
    train_f_df = make_feature(train_df)
    print(train_f_df.head(5))



if __name__ == '__main__':
    # make_npy()
    # make_only_increased()
    # make_period(5)
    # make_period(15)
    make_period_with_feature()