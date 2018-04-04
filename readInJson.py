# -*- coding: utf-8 -*-
# @Author: yanxia
# @Date:   2016-11-07 15:49:57
# @Last Modified by:   yanxia
# @Last Modified time: 2017-04-14 11:13:54

import json
import pandas as pd
import numpy as np
from datetime import datetime
# import ijson


def load_json(all_files):
    d = []
    for f in all_files:
        # print "********************Data Info Begin************************"
        # print f
        with open(f) as json_data:
            currentDF = pd.DataFrame(json.loads(l) for l in json_data)
            d.append(currentDF)
        #   d = json.load(json_data)
        # print currentDF.info()
        # print d.columns.values
        # print "The data contains " + str(len(currentDF)) + " events and " + str(len(currentDF.columns)) + " columns"
        # print currentDF.columns
        # print "The first line of information"
        # print currentDF[:10]
        # print currentDF.tail()

        # print "********************Data Info End************************"

    return d

# Convert timestamp to timedate object: 2016-09-23T17:09:21.000Z
def extract_timestamp(d):
    d_new = d
    d_new['timestamp'] = d_new['timestamp'].apply(lambda x: x.get('$date'))
    d_new['datetime'] = d_new['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))
    return d_new

# Convert timestamp to timedate object: 2016-09-23T17:09:21.000Z
def extract_timestamp_TI(d):
    # Remove the empty message
    d_new = d[d.timestamp != 'undefined']
    # d_new['datetime'] = d_new['timestamp']
    # d_new['datetime'] = pd.to_datetime(d_new['timestamp'], format = '%Y-%m-%dT%H:%M:%S.%fZ')
    d_new['timestamp'] = d_new['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))
    # print "Finished converting timestamp"
    return d_new


def extract_id(d):
    d_new = d
    d_new['_id'] = d_new['_id'].apply(lambda x: x.get('$oid'))
    return d_new


def generate_sensorvariables(sensors):
    # Create dictionary for sensor readings
    # [u'SS:energy:1' u'MS:movement:1' u'MS:temperature:1' u'MS:light:1' u'MS:humidity:1' u'MS:uv:1']

    sensorReading = {}
    for s in sensors:
        if 'energy' in s:
            sensorReading[s] = 'Watts'

        if 'movement' in s:
            sensorReading[s] = 'sl_Alarm'

        if 'temperature' in s:
            sensorReading[s] = 'CurrentTemperature'

        if 'light' in s:
            sensorReading[s] = 'CurrentLevel'

        if 'humidity' in s:
            sensorReading[s] = 'CurrentLevel'

        if 'uv' in s:
            sensorReading[s] = 'CurrentLevel'
    """
    print s
    print "----"
    print sensorReading[s]

    sensorReading['MS:Movement:1'] = 'sl_Alarm' # TAMPER_ALARM  # TRIP_ALARM  # UNTRIP_ALARM
    sensorReading['MS:temperature:1'] = 'CurrentTemperature'
    sensorReading['MS:light:1'] = 'CurrentLevel'
    sensorReading['MS:humidity:1'] = 'CurrentLevel'
    sensorReading['MS:uv:1'] = 'CurrentLevel'

    sensorReading[sensors[0]] = 'Watts'
    sensorReading[sensors[1]] = 'sl_Alarm'
    sensorReading[sensors[2]] = 'CurrentTemperature'
    sensorReading[sensors[3]] = 'CurrentLevel'
    sensorReading[sensors[4]] = 'CurrentLevel'
    if len(sensors) > 5:
        sensorReading[sensors[5]] = 'CurrentLevel'
    """
    return sensorReading


def generate_sensorvariables_TI(sensors):
    # Create dictionary for sensor readings
    # [u'SS:energy:1' u'MS:movement:1' u'MS:temperature:1' u'MS:light:1' u'MS:humidity:1' u'MS:uv:1']

    sensorReading = {}
    for s in sensors:
        if 'rssi' in s:
            sensorReading[s] = 'rssi'

        if 'pressure' in s:
            sensorReading[s] = 'pressure'

        if 'temperature' in s:
            sensorReading[s] = 'temperature'

        if 'lux' in s:
            sensorReading[s] = 'lux'

        if 'humidity' in s:
            sensorReading[s] = 'humidity'

    """
    print s
    print "----"
    print sensorReading[s]

    sensorReading['MS:Movement:1'] = 'sl_Alarm' # TAMPER_ALARM  # TRIP_ALARM  # UNTRIP_ALARM
    sensorReading['MS:temperature:1'] = 'CurrentTemperature'
    sensorReading['MS:light:1'] = 'CurrentLevel'
    sensorReading['MS:humidity:1'] = 'CurrentLevel'
    sensorReading['MS:uv:1'] = 'CurrentLevel'

    sensorReading[sensors[0]] = 'Watts'
    sensorReading[sensors[1]] = 'sl_Alarm'
    sensorReading[sensors[2]] = 'CurrentTemperature'
    sensorReading[sensors[3]] = 'CurrentLevel'
    sensorReading[sensors[4]] = 'CurrentLevel'
    if len(sensors) > 5:
        sensorReading[sensors[5]] = 'CurrentLevel'
    """
    return sensorReading
