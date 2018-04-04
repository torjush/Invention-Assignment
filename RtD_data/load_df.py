import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import glob
import readInJson
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix


pd.options.mode.chained_assignment = None


def load_all_data(folder, file_names):
    all_files = glob.glob(folder + file_names)

    print('Loading data')
    all_data = readInJson.load_json(all_files)

    print('All data loaded, concatenating data frames')
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
    indices = []
    for i in range(data.shape[0] - window_length):
        # Filter on movement data
        if np.abs(data[i:i + window_length][1].mean()) > filter_threshold:
            indices.append(i)

    vectors = np.ndarray(shape=(len(indices), data.shape[1] * window_length))
    for i, index in enumerate(indices):
        window = data[index:index + window_length, :]
        m = window.mean(axis=0)

        window = (window - m)
        vectors[i] = window.reshape(data.shape[1] * window_length)
    return vectors


def prepare_data(devices, data_columns, filter_threshold, window_length):
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


def grid_search_data_fields(devices, data_column_matrix, filter_thresholds, window_lengths):
    """Do grid search on all the configurations of data columns
    (humidity, temperature, acceleration, etc.), the filter_thresholds
    and the window lengths.
    """
    length = len(data_column_matrix) * len(filter_thresholds) * len(window_lengths)
    scores = np.ndarray(shape=(len(data_column_matrix), len(filter_thresholds), len(window_lengths)))
    printer = print_grid_search(length)
    next(printer)

    for i, data_columns in enumerate(data_column_matrix):
        for j, filter_threshold in enumerate(filter_thresholds):
            for k, window_length in enumerate(window_lengths):
                next(printer)
                X, y = prepare_data(devices, data_columns, filter_threshold, window_length)
                scores[i][j][k] = k_fold(X, y)

    try:
        next(printer)
    except StopIteration:
        pass
    return scores


def print_grid_search(length):
    """Print a status bar, since grid search can take a while"""
    print("Running Grid Search:")
    toolbar_width = 32
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1))
    yield

    for i in range(1, length+1):
        if (i % (length // toolbar_width)) == 0:
            sys.stdout.write("=")
            sys.stdout.flush()
        yield

    sys.stdout.write('\n')
    return


def k_fold(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)

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
    return score


def print_matrix(mat, xlabels=None, ylabels=None):
    if xlabels:
        sys.stdout.write("\t")
        for label in xlabels:
            sys.stdout.write("|" + str(label) + "\t")
        sys.stdout.write("|\n")
    for i in range(mat.shape[0]):
        if ylabels:
            sys.stdout.write(str(ylabels[i]) + "\t")
        for j in range(mat.shape[1]):
            if mat[i][j] == np.max(mat):
                sys.stdout.write("|\033[1;36m{:.4f}\033[0m\t".format(mat[i][j]))
            else:
                sys.stdout.write("|{:.4f}\t".format(mat[i][j]))
        sys.stdout.write("|\n")


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

    scores = grid_search_data_fields(fridge_data, data_column_matrix, filter_thresholds, window_lengths)
    print("Cross Validation result from training Gaussian Naive Bayes")
    for i in range(scores.shape[0]):
        print("With data columns: " + ", ".join(data_column_matrix[i]))
        print_matrix(scores[i][:][:])


if __name__ == '__main__':
    main()
