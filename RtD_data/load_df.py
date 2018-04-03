from collections import deque
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import os
import glob
import readInJson
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans


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

    return device_df.drop(['x', 'y', 'z'], axis=1)


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
    data = _filter(data, 5)
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
        if data[i][1] < threshold:
            del_indices.append(i)
    data = np.delete(data, del_indices, axis=0)
    return data


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
    rolling_rows = deque(maxlen=10)

    for timestamp, values in device_df.iterrows():
        rolling_rows.append(values.isna().all())
        # end group creation
        if start_time is not None and all(rolling_rows):
            end_time = timestamp
            event = device_df[start_time:end_time]
            event = event.dropna(how='all')
            groups.append(event)

            start_time = None

        # start group creation
        elif start_time is None and not all(rolling_rows):
            start_time = timestamp

    return groups


def k_fold(X, y):
    kf = KFold(n_splits=10, shuffle=True)

    avg = []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        score = accuracy_score(y_true=y_test, y_pred=y_pred)
        avg.append(score)
        print("Accuracy:", score)
        print("Confusion matrix:")
        print_confusion_matrix(y_test, y_pred, ["Home 1", "Home 2", "Home 3"])
        print("\n\n")

    score = sum(avg) / len(avg)
    print("Accuracy of Naive Bayes: {:.4f}".format(score))


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
        'Remote Control': '247189ea0782',
        'Rope on Stairs': '247189e74381',
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

    rope = get_device_data(df, devices['Rope on Stairs'])
    rope = extract_imu(rope)
    rope = rope[data_columns[1:]]

    events = group_by_event(rope)
    print "Number of events:", len(events)

    #X = vectorize_and_filter(rope.values, window_length=50)

    k_means = KMeans(2)
    k_means.fit(X)


    vectors = k_means.cluster_centers_
    labels = k_means.labels_
    print labels



if __name__ == '__main__':
    main()
