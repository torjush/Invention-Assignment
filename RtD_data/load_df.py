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


def preprocess_imu(device_df, window_length='1s', threshold=2):
    """Returns windowed imu data from device_df,
    getting rid of NaN values, as these only occur when
    the environment measurements are out of sync with imu.
    If threshold is set, only windows with this number or
    more data points are returned"""
    device_df.set_index('timestamp', inplace=True)
    device_df.sort_index(inplace=True)

    imu_data = device_df[['x', 'y', 'z']].dropna(axis=0, how='all')
    windowed_imu = imu_data.rolling(window_length)
    imu_data = imu_data[windowed_imu.count() > threshold]
    return imu_data


def group_by_event(device_df):
    """Returns a list of groups
    """
    groups = []

    start_time = None
    xs, ys, zs = [], [], []

    for timestamp, values in device_df.iterrows():
        # end group creation
        if start_time != None and values.isna().any():
            end_time = timestamp
            event = device_df[start_time:end_time]
            groups.append(event)
            start_time = None

        # start group creation
        elif start_time == None and not values.isna().any():
            start_time = timestamp

    return groups


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

    objects01 = {
        '247189e98685': 'Remote Control',
        '247189e83001': 'Spider Stick',
        '247189e72603': 'Garden Door',
        '247189e78180': 'Fridge',
        '247189e76106': 'Breakfast Chair',
        '247189e87d85': 'Tray',
    }
    objects02 = {
        '247189e98d83': 'Chair Pillow',
        '247189ea0782': 'Remote Control',
        '247189e74381': 'Rope on Stairs',
        '247189e64706': 'Kitchen Drawer',
        '247189e61784': 'Fridge'
    }

    objects03 = {
        '247189e61802': 'Kitchen Chair',
        '247189e61682': 'Fridge',
        '247189e76c05': 'Remote Control',
        '247189e88b80': 'Kitchen Cabinet Door',
        '247189e8e701': 'Knitting Needle',
        '247189e6c680': 'Tablet'
    }
    plot_objects(objects02, df)


if __name__ == '__main__':
    main()
