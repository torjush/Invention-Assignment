# -*- coding: utf-8 -*-
# @Author: yanxia
# @Date:   2017-03-30 16:23:44
# @Last Modified by:   yanxia
# @Last Modified time: 2017-10-16 09:08:08

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import datetime
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
matplotlib.style.use('ggplot')

import preprocess
# import cluster

sensors = ['accelx', 'accely', 'accelz', 
           'magx', 'magy', 'magz', 
           'gyrox', 'gyroy', 'gyroz', 
           'rssi_air', 'pressure_air', 'humidity_air', 
           'temperature_air', 'lux_air', 'battery_air']


objects01 = { '247189e98685': 'Remote Control',
              '247189e83001': 'Spider Stick',
              '247189e72603': 'Garden Door',
              '247189e78180': 'Fridge',
              '247189e76106': 'Breakfast Chair',
              '247189e87d85': 'Tray',
}

objects02 = { '247189e98d83': 'Chair Pillow',
              '247189ea0782': 'Remote Control',
              '247189e74381': 'Rope on Stairs',
              '247189e64706': 'Kitchen Drawer',
              '247189e61784': 'Fridge'
}

objects03 = { '247189e61802': 'Kitchen Chair',
              '247189e61682': 'Fridge',
              '247189e76c05': 'Remote Control',
              '247189e88b80': 'Kitchen Cabinet Door',
              '247189e8e701': 'Knitting Needle',
              '247189e6c680': 'Tablet'
}

objIndex = 2
objects = [objects01,objects02,objects03][objIndex]

def main():
    dataDir = '/field/'  #'/data/'
    fileName = "*.json"
    # fileName = "*07-*"  # "*2017-04-05.txt"
    numHome = 2  # Use everything is 'ALL'; otherwise integer
    numDevice = objects #'ALL'  # Use everything is 'ALL'; otherwise integer
    numModality = [0, 10,11,12]#'ALL'  # Use everything is 'ALL'; otherwise integer      
    showPLOT = True
    isNORMALIZED = True
    startDate = datetime.date(2017, 9, 1)
    endDate = datetime.date(2017, 9, 29)

    alldata = {}
    alldata, perDevice = preprocess.convert_to_DataFrame_TI(dataDir, fileName, numHome, numDevice, numModality, showPLOT, startDate, endDate)


if __name__ == "__main__":main()
