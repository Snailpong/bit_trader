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


if __name__ == '__main__':
    train_x_array = get_array('./data/train_x_df.csv')
    train_y_array = get_array('./data/train_y_df.csv')
    test_x_array = get_array('./data/test_x_df.csv')

    # train_x_array (7661, 1380, 10), train_y_array (7661, 120, 10), test_y_array (535, 120, 10)

    np.save('./data/train_x', train_x_array)
    np.save('./data/train_y', train_y_array)
    np.save('./data/test_x', test_x_array)