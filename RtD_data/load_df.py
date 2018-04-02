import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import readInJson
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering


def load_all_data(folder, file_names):
    all_files = glob.glob(folder + file_names)

    print('=== Loading data ===')
    all_data = readInJson.load_json(all_files)

    print('=== All data loaded, concatenating data frames ===')
    df = pd.concat(all_data)
    df = readInJson.extract_timestamp_TI(df)
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)
    return df


def get_device_data(df, device_id):
    device_df = df[df['deviceId'] == device_id]
    device_df = device_df.join(pd.DataFrame(device_df["data"].to_dict()).T)
    device_df.drop('data', axis=1, inplace=True)

    return device_df


def threshold_data(device_df, keep_values, window_length='10s', threshold=2):
    """Returns windowed imu data from device_df,
    getting rid of NaN values, as these only occur when
    the environment measurements are out of sync with imu.
    If threshold is set, only windows with this number or
    more data points are returned"""
    imu_data = device_df[keep_values].dropna(axis=0, how='all')
    windowed_imu = imu_data.rolling(window_length)
    imu_data = imu_data[windowed_imu.count() > threshold]
    return imu_data


def extract_imu(device_df):
    device_df[['gyro_x', 'gyro_y', 'gyro_z']] = device_df[['x', 'y', 'z']].where(device_df['event'] == 'gyro')
    device_df[['accel_x', 'accel_y', 'accel_z']] = device_df[['x', 'y', 'z']].where(device_df['event'] == 'accel')
    device_df[['mag_x', 'mag_y', 'mag_z']] = device_df[['x', 'y', 'z']].where(device_df['event'] == 'mag')

    device_df.drop(['x', 'y', 'z'], inplace=True)
    return device_df


def preprocess(df):
    if 'temperature' in df.columns:
        df['temperature'] = df['temperature'].fillna(method='bfill')
    if 'pressure' in df.columns:
        df['pressure'] = df['pressure'].fillna(method='bfill')
    if 'lux' in df.columns:
        df['lux'] = df['lux'].fillna(method='bfill')
    if 'humidity' in df.columns:
        df['humidity'] = df['humidity'].fillna(method='bfill')
    if 'gyro_x' in df.columns:
        df[['gyro_x', 'gyro_y', 'gyro_z']] = df[['gyro_x', 'gyro_y', 'gyro_z']].fillna(0)
    if 'accel_x' in df.columns:
        df[['accel_x', 'accel_y', 'accel_z']] = df[['accel_x', 'accel_y', 'accel_z']].fillna(0)
    if 'mag_x' in df.columns:
        df[['mag_x', 'mag_y', 'mag_z']] = df[['mag_x', 'mag_y', 'mag_z']].fillna(0)

    return df


def group_by_event(device_df):
    """Returns a list of groups
    """
    groups = []

    start_time = None

    for timestamp, values in device_df.iterrows():
        # end group creation
        if start_time is not None and values.isna().all():
            end_time = timestamp
            event = device_df[start_time:end_time]
            event = event.dropna()
            groups.append(event)
            start_time = None

        # start group creation
        elif start_time is None and not values.isna().all():
            start_time = timestamp

    return groups


def make_featurevectors(grouped_dfs, N_fft):
    feature_vectors = np.ndarray(shape=(len(grouped_dfs), 3*N_fft))
    for i, group in enumerate(grouped_dfs):
        x = group['x'].values
        y = group['y'].values
        z = group['z'].values

        X = np.fft.fft(x, N_fft)
        Y = np.fft.fft(y, N_fft)
        Z = np.fft.fft(z, N_fft)

        feature_vector = np.concatenate((X, Y, Z))
        feature_vector = np.abs(feature_vector)

        feature_vectors[i, :] = feature_vector
    return feature_vectors


def main():
    data_directory = '/field/'

    folder = os.getcwd() + data_directory
    files = "*.json"

    df = load_all_data(folder, files)

    devices = {
        'fridge_1': '247189e78180',
        'fridge_2': '247189e61784',
        'fridge_3': '247189e61682',
        'chair_1': '247189e76106',
        'chair_2': '247189e98d83',
        'chair_3': '247189e61802',
        'Remote Control': '247189ea0782'
    }
    device = get_device_data(df, devices['Remote Control'])

    device = extract_imu(device)

    data_columns = [
        'temperature',
        'accel_x',
        'accel_y',
        'accel_z',
        'gyro_x',
        'gyro_y',
        'gyro_z',
    ]

    data = preprocess(device[data_columns])

if __name__ == '__main__':
    main()
