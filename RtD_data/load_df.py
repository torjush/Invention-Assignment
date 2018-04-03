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


def extract_imu(device_df):
    if {'x', 'y', 'z'}.issubset(device_df.columns):
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


def vectorize_and_filter(data, window_length):
    """Window values in data and concatenate signal to make vectors"""
    data = _filter(data, 2)
    print("=== Vectorizing data ===")
    vectors = np.ndarray(
        shape=(data.shape[0] - window_length, data.shape[1] * window_length)
        )
    for i in range(data.shape[0] - window_length):
        vector = data[i:i + window_length, :].reshape(data.shape[1] * window_length)
        vectors[i] = vector

    return vectors


def _filter(data, threshold):
    print("=== Removing data with little movement ===")
    del_indices = []
    for i in range(data.shape[0]):
        if data[i][1].mean() < threshold:
            del_indices.append(i)
    data = np.delete(data, del_indices, axis=0)
    return data


def prepare_data(devices, data_columns):
    processed = {}
    for i, device in enumerate(devices):
        processed[device] = extract_imu(devices[device])
        processed[device] = preprocess(processed[device][data_columns])

        try:
            X_new = vectorize_and_filter(processed[device].values, window_length=50)
            y_new = np.ones(X_new.shape[0]) * device
            X = np.concatenate((X, X_new), axis=0)
            y = np.concatenate((y, y_new))
        except NameError:
            X = vectorize_and_filter(processed[device].values, window_length=50)
            y = np.ones(X.shape[0]) * device

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test


def print_confusion_matrix(y_true, y_pred, labels=None):
    cf = confusion_matrix(y_true, y_pred)
    if labels:
        pred_labels = ['Predicted ' + l for l in labels]
    df = pd.DataFrame(cf, index=labels, columns=pred_labels)
    print(df)


def grid_search_data_fields(devices, data_column_matrix):
    for data_columns in data_column_matrix:
        X_train, X_test, y_train, y_test = prepare_data(devices, data_columns)
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        score = accuracy_score(y_true=y_test, y_pred=y_pred)

        print("=== Training naive bayes classifier on the following data ===")
        print("=== " + ", ".join(data_columns) + " ===")

        print("Accuracy: {:.4f}".format(score))

        print("Confusion matrix:")
        print_confusion_matrix(y_test, y_pred, ["Home 1", "Home 2", "Home 3"])


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

    data_column_matrix = [
        ['humidity',
         'accel_x',
         'accel_y',
         'accel_z',
         'gyro_x',
         'gyro_y',
         'gyro_z'],
        ['temperature',
         'accel_x',
         'accel_y',
         'accel_z',
         'gyro_x',
         'gyro_y',
         'gyro_z'],
        ['lux',
         'accel_x',
         'accel_y',
         'accel_z',
         'gyro_x',
         'gyro_y',
         'gyro_z'],
        ['lux',
         'gyro_x',
         'gyro_y',
         'gyro_z']

    ]

    fridge_data = {
        1: get_device_data(df, devices['fridge_1']),
        2: get_device_data(df, devices['fridge_2']),
        3: get_device_data(df, devices['fridge_3'])
    }
    grid_search_data_fields(fridge_data, data_column_matrix)


if __name__ == '__main__':
    main()
