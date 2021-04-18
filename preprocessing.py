import numpy as np
import pandas as pd


def df2d_to_array3d(df_2d):
    feature_size = df_2d.iloc[:,2:].shape[1]
    time_size = len(df_2d.time.value_counts())
    sample_size = len(df_2d.sample_id.value_counts())
    sample_index = df_2d.sample_id.value_counts().index
    array_3d = df_2d.iloc[:,2:].values.reshape([sample_size, time_size, feature_size])
    
    return array_3d


def get_dataframe():
    train_x_df = pd.read_csv('./data/train_x_df.csv')
    train_y_df = pd.read_csv('./data/train_y_df.csv')

    return train_x_df, train_y_df


if __name__ == '__main__':
    train_x_df, train_y_df = get_dataframe()

    train_x_array = df2d_to_array3d(train_x_df)
    train_y_array = df2d_to_array3d(train_y_df)

    # train_x_array (7661, 1380, 10), train_y_array (7661, 120, 10)

    np.save('./data/train_x', train_x_array)
    np.save('./data/train_y', train_y_array)