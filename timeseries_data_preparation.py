import json
import os
import pickle
from datetime import datetime, timedelta

import pandas as pd
from sklearn import preprocessing

import math

import numpy as np
from sklearn.model_selection import train_test_split

import helper
# from helpers.data_generators import EmbeddingObjectSaver



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATETIME_PATTERN = "%Y-%m-%d %H:%M:%S"

parametersFilePath = "parameters/classify_chartevents_parameters.json"

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
all_features = features_chartevents
all_features.extend(features_labevents)



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
    while time_bucket <= patient['window_endtime']:
        timestamps = []
        for index in events.index:
            diff = time_bucket - index
            if diff.days == 0 and diff.seconds/60 < 60:
                if index not in buckets.keys():
                    timestamps.append(index)
        buckets[time_bucket] = events.loc[timestamps, :]
        time_bucket += timedelta(hours=1)
    events_buckets = pd.DataFrame({})
    for bucket in buckets.keys():
        events_in_bucket = dict()
        for column in events.columns:
            events_in_bucket[column] = np.max(buckets[bucket][column])
        events_in_bucket = pd.DataFrame(events_in_bucket, index=[bucket])
        events_buckets = pd.concat([events_buckets, events_in_bucket])
    events_buckets = events_buckets.sort_index()
    events_buckets = events_buckets.fillna(0)
    # Create data file from the buckets
    events_buckets.to_csv(parameters['dataPath'] + '{}.csv'.format(icustayid))