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
    return df


def get_device_data(df, device_id):
    device_df = df[df['deviceId'] == device_id]
    device_df = device_df.join(pd.DataFrame(device_df["data"].to_dict()).T)
    device_df.drop('data', axis=1, inplace=True)
    device_df = readInJson.extract_timestamp_TI(device_df)

    return device_df

def get_rolling_device(device):
    device.set_index('timestamp', inplace=True)

    device.sort_index(inplace=True)

    accel = device[['x', 'y', 'z']].dropna(axis=0, how='all') # drop NaN values
    return accel.rolling('30min') # rolling window by 30 minutes


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
    fridge_1 = get_rolling_device(device)
    fridge_1.mean().plot() # not very useful right now
    plt.show()

    # for device_name, device_id in devices.iteritems():
    #     device_df = get_device_data(df, device_id)
    #     print('Device: {}'.format(device_name))
    #     print(device_df.describe())



if __name__ == '__main__':
    main()
