import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import os
import sys
import glob
import readInJson
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, KFold
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
        tmp = df[['gyro_x', 'gyro_y', 'gyro_z']]
        tmp = tmp.fillna(0)
        df[['gyro_x', 'gyro_y', 'gyro_z']] = tmp
    if 'accel_x' in df.columns:
        tmp = df[['accel_x', 'accel_y', 'accel_z']]
        tmp = tmp.fillna(0)
        df[['accel_x', 'accel_y', 'accel_z']] = tmp
    if 'mag_x' in df.columns:
        tmp = df[['mag_x', 'mag_y', 'mag_z']]
        tmp = tmp.fillna(0)
        df[['mag_x', 'mag_y', 'mag_z']] = tmp

    return df


def vectorize_and_filter(data, filter_threshold, window_length):
    """Window values in data and concatenate signal to make vectors"""
    print("=== Vectorizing data ===")

    # setup toolbar
    toolbar_width = 10
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1))
    indices = []
    for i in range(data.shape[0] - window_length):
        if (i + 1) % ((data.shape[0] - window_length) // toolbar_width) == 0:
            sys.stdout.write("=")
            sys.stdout.flush()

        # Filter on movement data
        if np.abs(data[i:i + window_length][1].mean()) > filter_threshold:
            indices.append(i)

    vectors = np.ndarray(shape=(len(indices), data.shape[1] * window_length))
    for i, index in enumerate(indices):
        window = data[index:index + window_length, :]
        m = window.mean(axis=0)
        diff = np.max(window, axis=0) - np.min(window, axis=0)

        window = (window - m)
        vectors[i] = window.reshape(data.shape[1] * window_length)
    sys.stdout.write('\n')
    return vectors


def prepare_data(devices, filter_threshold, window_length, data_columns):
    processed = {}
    for device in devices:
        processed[device] = extract_imu(devices[device])
        processed[device] = preprocess(processed[device][data_columns])

        try:
            X_new = vectorize_and_filter(processed[device].values, filter_threshold, window_length)
            y_new = np.ones(X_new.shape[0]) * device
            X = np.concatenate((X, X_new), axis=0)
            y = np.concatenate((y, y_new))
        except NameError:
            X = vectorize_and_filter(processed[device].values, filter_threshold, window_length)
            y = np.ones(X.shape[0]) * device

    return X, y


def print_confusion_matrix(y_true, y_pred, labels=None):
    cf = confusion_matrix(y_true, y_pred)
    if labels:
        pred_labels = ['Predicted ' + l for l in labels]
        df = pd.DataFrame(cf, index=labels, columns=pred_labels)
    else:
        df = pd.DataFrame(cf)
    print(df)


def grid_search_data_fields(devices, filter_thresholds, window_lengths, data_column_matrix):
    scores = np.ndarray(shape=(len(filter_thresholds), len(window_lengths), len(data_column_matrix)))
    for i, filter_threshold in enumerate(filter_thresholds):
        for j, window_length in enumerate(window_lengths):
            for k, data_columns in enumerate(data_column_matrix):
                X, y = prepare_data(devices, filter_threshold, window_length, data_columns)
                scores[i][j][k] = k_fold(X, y)

    return scores


def k_fold(X, y):
    kf = KFold(n_splits=5, shuffle=True)

    avg = []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        score = accuracy_score(y_true=y_test, y_pred=y_pred)
        avg.append(score)

    score = sum(avg) / len(avg)
    print("Accuracy of Naive Bayes: {:.4f}".format(score))
    return score


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

    fridge_data = {
        1: get_device_data(df, devices['fridge_1']),
        2: get_device_data(df, devices['fridge_2']),
        3: get_device_data(df, devices['fridge_3'])
    }


    filter_thresholds = [1, 5, 10, 20]
    window_lengths = [10, 20, 50, 100]

    data_column_matrix = [
        ['humidity',
         'accel_x', 'accel_y', 'accel_z',
         'gyro_x', 'gyro_y', 'gyro_z'],
        ['temperature',
         'accel_x', 'accel_y', 'accel_z',
         'gyro_x', 'gyro_y', 'gyro_z'],
        ['lux',
         'accel_x', 'accel_y', 'accel_z',
         'gyro_x', 'gyro_y', 'gyro_z'],
        ['lux',
         'gyro_x', 'gyro_y', 'gyro_z']
    ]

    scores = grid_search_data_fields(fridge_data, filter_thresholds, window_lengths, data_column_matrix)
    print("=== Cross Validation result from training Gaussian Naive Bayes ===")
    for i in range(scores.shape[0]):
        print("=== With filter_threshold: {} ===".format(filter_thresholds[i]))
        for j in range(scores.shape[1]):
            print("=== With window_length: {} ===".format(window_lengths[j]))
            for k in range(scores.shape[2]):
                print("=== With data columns: ===")
                print("=== " + ", ".join(data_column_matrix[k]) + " ===")
                print("Accuracy: {:.4f}".format(scores[i, j, k]))


if __name__ == '__main__':
    main()
