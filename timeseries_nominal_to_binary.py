"""
Creates the binary columns for nominal data, adding the columns that doesn't appear at each patients events
"""
import os

import pandas as pd
import numpy as np

import helper

mimic_data_path = "/home/mattyws/Documents/mimic_data/"
events_files_path = mimic_data_path + 'data_organism_resistence/'
new_events_files_path = mimic_data_path + 'data_organism_resistence_binary/'
if not os.path.exists(new_events_files_path):
    os.mkdir(new_events_files_path)
# Get categorical valued events to transform to binary, labevents doens't have any categorical event
categorical_features_chartevents = ['chartevents' + '_' +itemid for itemid in helper.FEATURES_ITEMS_TYPE.keys()
                                    if helper.FEATURES_ITEMS_TYPE[itemid] == helper.CATEGORICAL_LABEL]

dataset_csv = pd.read_csv('dataset.csv')

# Collect nominal data from values and create the binary representation
nominal_events = dict()
for index, patient in dataset_csv.iterrows():
    print("===== {} =====".format(patient['hadm_id']))
    if os.path.exists(events_files_path + '{}.csv'.format(patient['hadm_id'])):
        # Get events and change nominal to binary
        events = pd.read_csv(events_files_path + '{}.csv'.format(patient['hadm_id']))
        nominal_in_events = [itemid for itemid in categorical_features_chartevents if itemid in events.columns]
        for itemid in nominal_in_events:
            if itemid not in nominal_events.keys():
                nominal_events[itemid] = set()
            nominal_events[itemid] |= set(events[itemid].unique())
        events = pd.get_dummies(events, columns=nominal_in_events, dummy_na=False)
        events.to_csv(new_events_files_path + '{}.csv'.format(patient['hadm_id']), index=False)

print("================ Add missing events ================")
# Now add for each patient the binary columns for the values that doesn't appear
for index2, patient in dataset_csv.iterrows():
    print("===== {} =====".format(patient['hadm_id']))
    if os.path.exists(events_files_path + '{}.csv'.format(patient['hadm_id'])):
        events = pd.read_csv(new_events_files_path + '{}.csv'.format(patient['hadm_id']))
        for itemid in nominal_events.keys():
            for value in nominal_events[itemid]:
                if value is np.nan:
                    continue
                if itemid + '_' + str(value) not in events.columns:
                    events.loc[:, itemid + '_' + str(value)] = pd.Series(np.zeros(len(events)), index=events.index)
        events.to_csv(new_events_files_path + '{}.csv'.format(patient['hadm_id']), index=False)