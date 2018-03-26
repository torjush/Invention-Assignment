import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import readInJson


def load_all_data(folder, file_names):
    all_files = glob.glob(folder + file_names)

    print('=== Loading data ===')
    all_data = readInJson.load_json(all_files)

    print('=== All data loaded, concatenating data frames ===')
    df = pd.concat(all_data)
    df = readInJson.extract_timestamp_TI(df)
    df.sort_values('timestamp', inplace=True)
    return df


def get_device_data(df, device_id):
    device_df = df[df['deviceId'] == device_id]
    device_df = device_df.join(pd.DataFrame(device_df["data"].to_dict()).T)
    device_df.drop('data', axis=1, inplace=True)

    return device_df


def preprocess_imu(device_df, window_length='20s', threshold=30):
    """Returns windowed imu data from device_df,
    getting rid of NaN values, as these only occur when
    the environment measurements are out of sync with imu.
    If threshold is set, only windows with this number or
    more data points are returned"""
    device_df.set_index('timestamp', inplace=True)
    device_df.sort_index(inplace=True)

    imu_data = device_df[['x', 'y', 'z']].dropna(axis=0, how='all')
    windowed_imu = imu_data.rolling(window_length)
    if threshold:
        imu_data[windowed_imu.count() > threshold].dropna(inplace=True)
    return imu_data


def main():
    data_directory = '/field/'

    folder = os.getcwd() + data_directory
    files = "*.json"

    df = load_all_data(folder, files)

    devices = {
        'fridge_1': '247189e78180',
        'fridge_2': '247189e61784',
        'fridge_3': '247189e61682',
    }

    device = get_device_data(df, devices['fridge_1'])
    fridge_1 = preprocess_imu(device)
    fridge_1.plot()  # not very useful right now
    plt.show()


if __name__ == '__main__':
    main()
