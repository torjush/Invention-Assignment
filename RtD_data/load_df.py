import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import readInJson

data_directory = '/field/'

folder = os.getcwd() + data_directory
files = "*.json"


all_files = glob.glob(folder + files)

all_data = readInJson.load_json(all_files)

df = pd.concat(all_data)

homes = df['homeId'].unique()
devices = df['deviceId'].unique()
sensors = df['event'].unique()
sensors = np.array([s.encode('ascii') for s in sensors])

example_fridge = df[df['deviceId'] == '247189e61784']
example_fridge = example_fridge.join(pd.DataFrame(example_fridge["data"].to_dict()).T)
example_fridge.drop('data', axis=1, inplace=True)

example_fridge = readInJson.extract_timestamp_TI(example_fridge)
example_fridge.set_index('timestamp', inplace=True)

print(example_fridge.head())

example_fridge[['x', 'y', 'z']].plot()
plt.show()
