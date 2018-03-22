# -*- coding: utf-8 -*-
# @Author: yanxia
# @Date:   2016-11-08 22:38:51
# @Last Modified by:   yanxia
# @Last Modified time: 2017-09-27 12:40:10

from __future__ import division
import readInJson
import glob
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from datetime import timedelta, datetime, date
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.style.use('ggplot')


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

def convert_to_DataFrame_TI(dataDir, fileName, numHome, numDevice, numModality, showPlot, startDate, endDate):
    # dataDir = '/data/'
    # fileName = "*0611.json"

    # Read in all Json data
    folder = os.getcwd() + dataDir
    all_files = glob.glob(folder + fileName)
    all_data = readInJson.load_json(all_files)
    print "Number of files in the folder: "
    print len(all_data)

    all_data = [readInJson.extract_timestamp_TI(i) for i in all_data]
    
    # # Preprocess first Json file.
    # d = all_data[0]

    # Preprocess all Json files
    d = pd.concat(all_data)

    homes = d['homeId'].unique()
    devices = d['deviceId'].unique()
    sensors = d['event'].unique()
    sensors = np.array([s.encode('ascii') for s in sensors])

    # Make new sensor list of seperated IMU X,Y,Z signals
    sensors = ['accelx', 'accely', 'accelz', 
               'magx', 'magy', 'magz', 
               'gyrox', 'gyroy', 'gyroz', 
               'rssi_air', 'pressure_air', 'humidity_air', 
               'temperature_air', 'lux_air', 'battery_air']
    IMU = sensors[:9]
    
    print sensors
    

    # # Merge multiple sensor data into one file per device
    sensorReadingColumns = ['data', 'timestamp']
    readings = {}
    combined = {}

    print "----------------- Logging devices information start -----------------"
    print "The number of homes: %d" %(len(homes)) 
    print homes
    print "The number of devices: %d" %(len(devices)) 
    print devices
    print "The number of sensors: %d" %(len(sensors)) 
    print sensors
    print "----------------- Logging devices information end -----------------"

    # Read in information for each device per sensor
    if showPlot:
        fig1 = plt.figure(1)
        fig2 = plt.figure(2)
        numBins = 30

    # Select number of devices and sensors
    if numHome is not "ALL":
        homes = homes[:numHome]

    if numDevice is not "ALL":
        devices = numDevice.keys()

    if numModality is not "ALL":
        sensors = [sensors[i] for i in numModality]
    
    # Retrieve sensor data
    deviceIndex = 0
    for device in devices:

        current = pd.DataFrame()
        currentReadings = pd.DataFrame()
 
        deviceIndex += 1
        sensorIndex = 0

        # Create a new figure for each device
        fig = plt.figure()

        for s in sensors:
            print "[Device , Sensor]: " + str(device) + " , " + str(s)

            temp = pd.DataFrame()
            # Seperate IMU sensors
            if s in IMU:
                # print '----------------IMU---------------' + s
                temp = d[(d.deviceId == device) & (d.event == s[:-1])][sensorReadingColumns]
                temp['data'] = temp['data'].apply(lambda x: x.get(s[-1]))
            else:
                # Store only timestamp and data information
                temp = d[(d.deviceId == device) & (d.event == s[-3:])][sensorReadingColumns]
                temp['data'] = temp['data'].apply(lambda x: x.get(s[:-4]))

            temp = temp.sort_values('timestamp')
            readings[(device, s)] = temp 

            currentReadings = temp.copy(deep=True)
            currentReadings.columns = [str(s), 'timestamp']
            
            currentReadings = currentReadings[['timestamp', str(s)]]
            

            if len(current) == 0:
                current = currentReadings
            else:
                current = pd.merge(current, currentReadings, how='outer', on=['timestamp'])
            print current.info()

            mydata = readings[(device, s)]
            mydata['data'] = mydata['data'].convert_objects(convert_numeric=True)
            
            print mydata.head()

            if showPlot and len(mydata) != 0:
                # Plot histogram data in Figure 1
                ax = fig1.add_subplot(len(sensors), len(devices), sensorIndex * len(devices) + deviceIndex)

                # Plot time series in Figure 2
                ax1 = fig.add_subplot(len(sensors), 1, sensorIndex+1)
                ax1.plot_date(mydata['timestamp'], mydata['data'], alpha=0.9)
                if numDevice is not "ALL":
                    ax1.set_xlabel("Device: " + str(numDevice[device]))# + " " + str(s), size=10)
                else:
                    ax1.set_xlabel("Device: " + str(device))# + " " + str(s), size=10)
                ax1.set_ylabel(str(s), size=10)
                ax1.set_xlim([startDate, endDate])
                ax1.xaxis.set_minor_locator(HourLocator(np.arange(0, 25, 6)))
                ax1.xaxis.set_major_formatter(DateFormatter('%m-%d:%H'))   # ax1.xaxis.set_major_locator(DayLocator())
                ax1.tick_params(axis='x', labelsize=7)

            sensorIndex += 1
            print 'Sensor: ', sensorIndex

        combined[device] = current
        print "*********** Combined sensor data start ***********"
        print combined[device].describe()
        print combined[device].head()
        print combined[device].tail()
        print "*********** Combined sensor data end ***********"

    print "----------------- Processed log information start -----------------"
    print "The number of processed logs devices: %d" %(len(combined)) 
    print "----------------- Processed log information end -----------------"

    if showPlot:
        # multipage('pics/objects02.pdf')
        plt.tight_layout()
        plt.show()

    return readings, combined
