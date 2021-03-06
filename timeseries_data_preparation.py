import json
import os
import pickle
import re
from collections import Counter
from datetime import datetime, timedelta

import pandas as pd
from sklearn import preprocessing

import math

import numpy as np
from sklearn.model_selection import train_test_split

import helper
# from helpers.data_generators import EmbeddingObjectSaver
labitems_prefix = 'labevents_'
items_prefix = 'chartevents_'
def transform_equal_columns(buckets):
    new_buckets = dict()
    for timestamp in buckets.keys():
        row = buckets[timestamp]
        for prefix in [items_prefix, labitems_prefix]:
            pairs = []
            if prefix == labitems_prefix:
                pairs = helper.ARE_EQUAL_LAB
            elif prefix == items_prefix:
                pairs = helper.ARE_EQUAL_CHART
            for pair in pairs:
                first_in_keys = prefix+pair[0]
                second_in_keys = prefix+pair[1]
                if first_in_keys in row.keys() and second_in_keys in row.keys():
                    row[first_in_keys].extend(row[second_in_keys])
                    row.pop(second_in_keys)
                elif first_in_keys not in row.keys() and second_in_keys in row.keys():
                    row[first_in_keys] = row[second_in_keys]
                    row.pop(second_in_keys)
        new_buckets[timestamp] = row
    return new_buckets

def preprocess_buckets(buckets, features_types):
    range_re = re.compile('\d+-\d+')
    number_plus = re.compile('\d+\+')
    # Loop all keys in the dictionary
    new_buckets = dict()
    for timestamp in buckets.keys():
        row = buckets[timestamp]
        for key in row.keys():
            for i in range(len(row[key])):
                if features_types[key] != helper.NUMERIC_LABEL:
                    continue
                if type(row[key][i]) == type(str()):
                    if row[key][i].lower() == 'notdone':
                        row[key][i] = 0
                    elif row[key][i].lower() == 'neg':
                        row[key][i] = -1
                    elif row[key][i].lower() == 'tr':
                        row[key][i] = np.nan
                    elif row[key][i].lower() == 'none':
                        row[key][i] = np.nan
                    elif len(row[key][i].strip() ) == 0:
                        row[key][i] = np.nan
                    elif row[key][i] == 'FEW':
                        row[key][i] = 1
                    elif row[key][i] == 'MOD':
                        row[key][i] = 2
                    elif row[key][i] == 'MANY':
                        row[key][i] = 3
                    elif range_re.match(row[key][i]):
                        numbers = re.findall('\d+', row[key][i])
                        numbers = [int(n) for n in numbers]
                        try:
                            row[key][i] = sum(numbers) / len(numbers)
                        except:
                            print(numbers)
                            print("erro no regex", row[key], row[key][i])
                    elif re.match('[-+]?\d*\.\d+|\d+ C', row[key][i]) :
                        numbers = re.findall('\d+', row[key][i])
                        numbers = [int(n) for n in numbers]
                        row[key][i] = numbers[0]
                    elif re.match('\d+\+', row[key][i]):
                        numbers = re.findall('\d+', row[key][i])
                        numbers = [int(n) for n in numbers]
                        row[key][i] = numbers[0]
                    elif row[key][i].startswith('LESS THAN') or row[key][i].startswith('<'):
                        numbers = re.findall('\d+', row[key][i])
                        if len(numbers) == 0:
                            row[key][i] = 0
                        else:
                            row[key][i] = float(numbers[0])
                    elif row[key][i].startswith('GREATER THAN') or row[key][i].startswith('>')\
                            or row[key][i].startswith('GREATER THEN'):
                        numbers = re.findall('\d+', row[key][i])
                        if len(numbers) == 0:
                            row[key][i] = 0
                        else:
                            row[key][i] = float(numbers[0])
                    elif row[key][i].startswith('EXCEEDS REFERENCE RANGE OF'):
                        numbers = re.findall('\d+', row[key][i])
                        if len(numbers) == 0:
                            row[key][i] = 0
                        else:
                            row[key][i] = float(numbers[0])
                    elif 'IS HIGHEST MEASURED PTT' in row[key][i]:
                        numbers = re.findall('\d+', row[key][i])
                        if len(numbers) == 0:
                            row[key][i] = 0
                        else:
                            row[key][i] = float(numbers[0])
                    elif 'UNABLE TO REPORT' in row[key][i] or 'VERIFIED BY REPLICATE ANALYSIS' in row[key][i]:
                        row[key][i] = np.nan
                    elif 'ERROR' in row[key][i] or 'UNABLE' in row[key][i]:
                        row[key][i] = np.nan
                    elif 'VERIFIED BY DILUTION' in row[key][i]:
                        row[key][i] = np.nan
                    else:
                        print(row[key][i], "====================================================")
                        row[key][i] = np.nan
                        continue
            row[key] = [value for value in row[key] if str(value) != 'nan']
        new_buckets[timestamp] = row
    return new_buckets


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATETIME_PATTERN = "%Y-%m-%d %H:%M:%S"

parametersFilePath = "parameters/data_parameters.json"

#Loading parameters file
print("========= Loading Parameters")
parameters = None
with open(parametersFilePath, 'r') as parametersFileHandler:
    parameters = json.load(parametersFileHandler)
if parameters is None:
    exit(1)

dataset_patients = pd.read_csv(parameters['datasetCsvFilePath'])
dataset_patients.loc[:,'admittime'] = pd.to_datetime(dataset_patients['admittime'], format=DATETIME_PATTERN)
dataset_patients.loc[:,'intime'] = pd.to_datetime(dataset_patients['intime'], format=DATETIME_PATTERN)
dataset_patients.loc[:,'window_starttime'] = pd.to_datetime(dataset_patients['window_starttime'], format=DATETIME_PATTERN)
dataset_patients.loc[:,'window_endtime'] = pd.to_datetime(dataset_patients['window_endtime'], format=DATETIME_PATTERN)
features_chartevents = ['chartevents_'+key for key in list(helper.FEATURES_ITEMS_LABELS.keys())]
features_labevents = ['labevents_'+key for key in list(helper.FEATURES_LABITEMS_LABELS.keys())]
# all_features = features_chartevents
# all_features.extend(features_labevents)

all_features, features_types = helper.get_attributes_from_arff(parameters['parametersArffFile'])
print(parameters['parametersArffFile'], parameters['dataPath'], parameters['dataPathBinary'])



if not os.path.exists(parameters['dataPath']):
    os.mkdir(parameters['dataPath'])

le = preprocessing.LabelEncoder()
le.fit(dataset_patients['class'].tolist())
# print(le.classes_)
classes = dataset_patients['class'].tolist()

# print(" ======= Selecting random sample ======= ")
# dataTrain, dataTest, labelsTrain, labelsTest = train_test_split(dataset_patients['hadm_id'].tolist(), classes,
#                                                                 stratify=classes, test_size=0.8)
dataTrain = dataset_patients['icustay_id']
labelsTrain = classes
for icustayid, icustay_class in zip(dataTrain, labelsTrain):
    print("===== {} =====".format(icustayid))
    if not os.path.exists(parameters['datasetFilesPath']+'{}.csv'.format(icustayid)):
        continue
    # Get patient row from dataset csv
    patient = dataset_patients[dataset_patients['icustay_id'] == icustayid].iloc[0]
    # Loading events
    events = pd.read_csv(parameters['datasetFilesPath'] +'{}.csv'.format(icustayid))
    events.loc[:,'event_timestamp'] = pd.to_datetime(events['event_timestamp'], format=DATETIME_PATTERN)
    # The data representation is the features ordered by id
    events = events.set_index(['event_timestamp']).sort_index()
    events = events.ffill()
    # Now add the events that doesn't appear with as empty column
    itemids_not_in_events = [itemid for itemid in all_features if itemid not in events.columns]
    for itemid in itemids_not_in_events:
        empty_events = np.empty(len(events))
        empty_events[:] = np.nan
        events[itemid] = pd.Series(empty_events, index=events.index)
    # Filtering
    events = events[[itemid for itemid in events.columns if '_'.join(itemid.split('_')[0:2]) in all_features]]
    time_bucket = patient['window_starttime']
    buckets = dict()
    while time_bucket < patient['window_endtime']:
        timestamps = []
        for index in events.index:
            diff = index - time_bucket
            if diff.days == 0 and diff.seconds/60 < 60:
                if index not in buckets.keys():
                    timestamps.append(index)
        buckets[time_bucket] = events.loc[timestamps, :].to_dict('list')
        time_bucket += timedelta(hours=1)
    buckets = transform_equal_columns(buckets)
    buckets = preprocess_buckets(buckets, features_types)
    events_buckets = pd.DataFrame({})
    for bucket in buckets.keys():
        events_in_bucket = dict()
        for column in buckets[bucket].keys():
            feature_type = features_types[column]
            # print(column, feature_type)
            if feature_type == helper.NUMERIC_LABEL:
                if len(buckets[bucket][column]) > 0:
                    try:
                        events_in_bucket[column] = np.nanmax(buckets[bucket][column])
                        if events_in_bucket[column] is None:
                            events_in_bucket[column] = np.nan
                    except Exception as e:
                        print(e)
                        print(buckets[bucket][column])
                        exit()
                else:
                    events_in_bucket[column] = np.nan
            elif feature_type == helper.CATEGORICAL_LABEL:
                if len(buckets[bucket][column]) > 0:
                    events_in_bucket[column] = Counter(buckets[bucket][column]).most_common(1)[0][0]
                else:
                    events_in_bucket[column] = np.nan
        events_in_bucket = pd.DataFrame(events_in_bucket, index=[bucket])
        events_buckets = pd.concat([events_buckets, events_in_bucket])
    events_buckets = events_buckets.sort_index()

    float_cols = [itemd for itemd in features_types.keys() if itemd in events_buckets.columns and
                  features_types[itemd] == helper.NUMERIC_LABEL]#events.select_dtypes(include=['float64']).columns
    # str_cols = events.select_dtypes(include=['object']).columns
    events.loc[:, float_cols] = events.loc[:, float_cols].fillna(0)
    # events.loc[:, str_cols] = events.loc[:, str_cols].fillna('Not Measured')
    # events_buckets = events_buckets.fillna(0)
    # Create data file from the buckets
    events_buckets.to_csv(parameters['dataPath'] + '{}.csv'.format(icustayid))