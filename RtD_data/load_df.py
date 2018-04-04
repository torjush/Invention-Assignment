from collections import deque
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
from matplotlib import cm
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
            start_time = None
            if len(event) > 60:
                groups.append(event)

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


def plot_objects(objects, df):
    fig, axes = plt.subplots(nrows=len(objects), sharex=True)
    i = 0
    for obj in objects:
        device = get_device_data(df, obj)
        data = preprocess_imu(device)
        print("=== plotting data for {} ===".format(objects[obj]))
        data.plot(ax=axes[i])
        axes[i].set_title(objects[obj])
        i += 1

    plt.show()


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
        'accel_x',
        'accel_y',
        'accel_z',
        'gyro_x',
        'gyro_y',
        'gyro_z',
    ]

    rope = get_device_data(df, devices['Rope on Stairs'])
    rope = extract_imu(rope)
    rope = rope[data_columns]

    events = group_by_event(rope)
    print "Number of events:", len(events)
    print "Avg length:", sum([len(event) for event in events])/len(events)
    print "std: ", np.std([len(event) for event in events])
    print "median: ", np.median([len(event) for event in events])

    #X = vectorize_and_filter(rope.values, window_length=50)

    features = np.ndarray(shape=(len(events),len(data_columns)*50))
    for j, event in enumerate(events):
        feature = np.empty(len(data_columns)*50)
        for i, column in enumerate(data_columns):
            df = event[[column]].dropna() # because we only look at one column, we have NaN's
            df = df.resample('200ms').mean().fillna(method='bfill') # resample
            vals = np.hstack(df.values) # df seems to give a list of lists
            vals = np.multiply(np.hamming(len(vals)), vals) # hamming window
            transformed = np.fft.fft(vals, n=50) # take the fft with 50 values
            feature[i*50:i*50+50] = np.abs(transformed)
        features[j] = feature


    k_means = KMeans(3)
    k_means.fit(features)


    def plot_events(events, labels, data_columns):
        ax = None
        for i, event in enumerate(events):
            #index = np.array(event.index.to_pydatetime(), dtype=np.datetime64)
            #values = np.hstack(event[[data_columns[0]]].values)
            #ind = np.concatenate([values, index])
            #new_events.append(ind)
            part = event[[data_columns[3]]]
            if ax is None:
                ax = part.plot()
            else:
                part.plot(ax=ax)
        colormap = [[cm.viridis, cm.plasma, cm.inferno][i] for i in labels]
        #line = LineCollection(new_events, cmap=ListedColormap(colormap))
        #fig1 = plt.figure()
        #plt.gca().add_collection(line)
        plt.show()


    vectors = k_means.cluster_centers_
    labels = k_means.labels_
    plot_events(events, labels, data_columns)
    print labels


if __name__ == '__main__':
    main()
