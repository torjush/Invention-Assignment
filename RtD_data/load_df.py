import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import readInJson
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


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

    df = filter_by_movement(df, '10s', 5)

    return df


def vectorize(data, window_length):
    """Window values in data and concatenate signal to make vectors"""
    vectors = np.ndarray(
        shape=(data.shape[0] - window_length, data.shape[1] * window_length)
        )
    for i in range(data.shape[0] - window_length):
        vector = data[i:i + window_length, :].reshape(data.shape[1] * window_length)
        vectors[i] = vector

    return vectors


def filter_by_movement(data_df, window_length, threshold):
    rolling_mean = data_df.rolling(window_length).mean()

    data_df = data_df.drop(data_df.where(rolling_mean['accel_x'] < threshold))
    return data_df


def print_confusion_matrix(y_true, y_pred, labels=None):
    cf = confusion_matrix(y_true, y_pred)
    if labels:
        pred_labels = ['Predicted ' + l for l in labels]
    df = pd.DataFrame(cf, index=labels, columns=pred_labels)
    print(df)


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

    data_columns = [
        'humidity',
        'accel_x',
        'accel_y',
        'accel_z',
        'gyro_x',
        'gyro_y',
        'gyro_z',
    ]

    fridge_data = {
        1: get_device_data(df, devices['fridge_1']),
        2: get_device_data(df, devices['fridge_2']),
        3: get_device_data(df, devices['fridge_3'])
    }

    for i, fridge in enumerate(fridge_data):
        fridge_data[fridge] = extract_imu(fridge_data[fridge])
        fridge_data[fridge] = preprocess(fridge_data[fridge][data_columns])

        try:
            X_new = vectorize(fridge_data[fridge].values, window_length=50)
            y_new = np.ones(X_new.shape[0]) * fridge
            X = np.concatenate((X, X_new), axis=0)
            y = np.concatenate((y, y_new))
        except NameError:
            X = vectorize(fridge_data[fridge].values, window_length=50)
            y = np.ones(X.shape[0]) * fridge

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    score = accuracy_score(y_true=y_test, y_pred=y_pred)

    print("Accuracy of Naive Bayes: {:.4f}".format(score))

    print("Confusion matrix:")
    print_confusion_matrix(y_test, y_pred, ["Home 1", "Home 2", "Home 3"])


if __name__ == '__main__':
    main()
