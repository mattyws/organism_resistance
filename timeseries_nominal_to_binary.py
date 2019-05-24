"""
Creates the binary columns for nominal data, adding the columns that doesn't appear at each patients events
"""
import os
from functools import partial

import pandas as pd
import numpy as np
import multiprocessing as mp
from pandas._libs import json

import helper

parametersFilePath = "parameters/data_parameters.json"

#Loading parameters file
print("========= Loading Parameters")
parameters = None
with open(parametersFilePath, 'r') as parametersFileHandler:
    parameters = json.load(parametersFileHandler)
if parameters is None:
    exit(1)

mimic_data_path = parameters['mimicDataPath']
events_files_path = parameters['dataPath']
new_events_files_path = parameters['dataPathBinary']
if not os.path.exists(new_events_files_path):
    os.mkdir(new_events_files_path)
# Get categorical valued events to transform to binary, labevents doens't have any categorical event
# categorical_features_chartevents = ['chartevents' + '_' +itemid for itemid in helper.FEATURES_ITEMS_TYPE.keys()
#                                     if helper.FEATURES_ITEMS_TYPE[itemid] == helper.CATEGORICAL_LABEL]
all_features, features_types  = helper\
    .get_attributes_from_arff(parameters['parametersArffFile'])
categorical_features_chartevents = [itemid for itemid in features_types.keys()
                                    if features_types[itemid] == helper.CATEGORICAL_LABEL]
print(len(features_types.keys()), len(categorical_features_chartevents))
dataset_csv = pd.read_csv('dataset.csv')

def binarize_nominal_events(icustay_id, categorical_events, events_files_path, new_events_files_path):
    nominal_events = []
    print("#### {} ####".format(icustay_id))
    if os.path.exists(events_files_path + '{}.csv'.format(icustay_id)):
        # Get events and change nominal to binary
        events = pd.read_csv(events_files_path + '{}.csv'.format(icustay_id))
        if 'Unnamed: 0' in events.columns:
            events = events.drop(columns=['Unnamed: 0'])
        nominal_in_events = [itemid for itemid in categorical_events if itemid in events.columns]
        # for itemid in nominal_in_events:
        #     if itemid not in nominal_events.keys():
        #         nominal_events[itemid] = set()
        #     nominal_events[itemid] |= set(events[itemid].dropna().unique())
        events = pd.get_dummies(events, columns=nominal_in_events, dummy_na=False)
        nominal_events = events.columns
        events.to_csv(new_events_files_path + '{}.csv'.format(icustay_id), index=False)
    print("#### End {} ####".format(icustay_id))
    return nominal_events

def fill_missing_events(icustay_id, all_features, new_events_files_path):
    print("#### {} ####".format(icustay_id))
    if os.path.exists(new_events_files_path + '{}.csv'.format(icustay_id)):
        events = pd.read_csv(new_events_files_path + '{}.csv'.format(icustay_id))
        events = events.fillna(0)
        if 'Unnamed: 0' in events.columns:
            events = events.drop(columns=['Unnamed: 0'])
        zeros = np.zeros(len(events))
        features_not_in = all_features - set(events.columns)
        for itemid in features_not_in:
            events.loc[:, itemid] = pd.Series(zeros, index=events.index)
        events = events.sort_index(axis=1)
        events.to_csv(new_events_files_path + '{}.csv'.format(icustay_id), index=False)
        print("#### End {} ####".format(icustay_id))


# Creating a partial maintaining some arguments with fixed values
partial_binarize_nominal_events = partial(binarize_nominal_events, categorical_events = categorical_features_chartevents,
                                          events_files_path=events_files_path, new_events_files_path=new_events_files_path)
# Using as arg only the icustay_id, bc of fixating the others parameters
args = list(dataset_csv['icustay_id'])
# If the dir already exists and it has files for all dataset already created, only loop to get all possible events
if os.path.exists(new_events_files_path) and len(os.listdir(new_events_files_path)) == len(dataset_csv):
    print("{} already created and have files for all dataset, fetching the columns")
    file_list = [new_events_files_path + x for x in os.listdir(new_events_files_path)]
    features_after_binarized = set()
    i = 0
    for file in file_list:
        if i % 1000 == 0:
            print("Read {} files".format(i))
        with open(file, 'r') as f:
            features_after_binarized |= set(f.readline().split(','))
        i += 1
else:
    # If doesn't exist, go binarize the values
    # The results of the processes
    results = []
    # Creating the pool
    with mp.Pool(processes=3) as pool:
        results = pool.map(partial_binarize_nominal_events, args)
    print("========== Get new features after the dummies ==========")
    features_after_binarized = set()
    for result in results:
        features_after_binarized |= set(result)
    # features_after_binarized = list(features_after_binarized)

print("========== Filling events ==========")
partial_fill_missing_events = partial(fill_missing_events, all_features=features_after_binarized,
                                      new_events_files_path=new_events_files_path)
with mp.Pool(processes=3) as pool:
    pool.map(partial_fill_missing_events, args)

# exit()
# for index, patient in dataset_csv.iterrows():
#     print("===== {} =====".format(patient['icustay_id']))
#     if os.path.exists(events_files_path + '{}.csv'.format(patient['icustay_id'])):
#         # Get events and change nominal to binary
#         events = pd.read_csv(events_files_path + '{}.csv'.format(patient['icustay_id']))
#         if 'Unnamed: 0' in events.columns:
#             events = events.drop(columns=['Unnamed: 0'])
#         nominal_in_events = [itemid for itemid in categorical_features_chartevents if itemid in events.columns]
#         for itemid in nominal_in_events:
#             if itemid not in nominal_events.keys():
#                 nominal_events[itemid] = set()
#             nominal_events[itemid] |= set(events[itemid].dropna().unique())
#         events = pd.get_dummies(events, columns=nominal_in_events, dummy_na=False)
#         events.to_csv(new_events_files_path + '{}.csv'.format(patient['icustay_id']), index=False)
#
# print("================ Add missing events ================")
# # Now add for each patient the binary columns for the values that doesn't appear
# for index2, patient in dataset_csv.iterrows():
#     print("===== {} =====".format(patient['icustay_id']))
#     if os.path.exists(new_events_files_path + '{}.csv'.format(patient['icustay_id'])):
#         events = pd.read_csv(new_events_files_path + '{}.csv'.format(patient['icustay_id']))
#         if 'Unnamed: 0' in events.columns:
#             events = events.drop(columns=['Unnamed: 0'])
#         for itemid in nominal_events.keys():
#             for value in nominal_events[itemid]:
#                 if value is np.nan:
#                     continue
#                 if itemid + '_' + str(value) not in events.columns:
#                     events.loc[:, itemid + '_' + str(value)] = pd.Series(np.zeros(len(events)), index=events.index)
#         events = events.fillna(0)
#         events.to_csv(new_events_files_path + '{}.csv'.format(patient['icustay_id']), index=False)