import json
import os
from functools import partial

import pandas as pd
import multiprocessing as mp

import helper

def filter_events_features(icustay_id, features_from_arffs, events_files_path):
    if os.path.exists(events_files_path + '{}.csv'.format(icustay_id)):
        print("#### {} ####".format(icustay_id))
        # Get events and change nominal to binary
        events = pd.read_csv(events_files_path + '{}.csv'.format(icustay_id))
        columns = events.columns
        for key in features_from_arffs.keys():
            filtered_columns = [column for column in columns if '_'.join(column.split('_')[:2]) in features_from_arffs[key]]
            new_events = events[filtered_columns]
            new_events.to_csv('{}/{}.csv'.format(key, icustay_id))
        print("#### End {} ####".format(icustay_id))


parametersFilePath = "parameters/data_parameters.json"
print("========= Loading Parameters")
parameters = None
with open(parametersFilePath, 'r') as parametersFileHandler:
    parameters = json.load(parametersFileHandler)
if parameters is None:
    exit(1)

mimic_data_path = parameters['mimicDataPath']
events_files_path = parameters['dataPathBinary']

arff_file_paths = [
    'arffs/dataset_organism_resistance_manualRemove.arff',
    'arffs/dataset_organism_resistance_manualRemove_IG.arff',
    'arffs/dataset_organism_resistance_manualRemove_noUseless.arff',
    'arffs/dataset_organism_resistance_manualRemove_noUseless_wrapper.arff',
    'arffs/dataset_organism_resistance_noUseless.arff',
    'arffs/dataset_organism_resistance_noUseless_wrapper.arff',
    'arffs/dataset_organism_resistance_IG.arff',
]

# Get features from arffs
print("========= Get features from arffs ===========")
features_from_arffs = dict()
for arff_file in arff_file_paths:
    file_name = arff_file.split('/')[1].split('.')[0]
    if not os.path.exists(file_name):
        os.mkdir(file_name)
    features_from_arffs[file_name] = helper.get_attributes_from_arff(arff_file)[0]
    features_from_arffs[file_name] = [x for x in features_from_arffs[file_name] if x.startswith('chartevents') or x.startswith('labevents')]
# Read dataset file
dataset_df = pd.read_csv('dataset.csv')

args = list(dataset_df['icustay_id'])
partial_filter_events_features = partial(filter_events_features, features_from_arffs=features_from_arffs,
                                         events_files_path=events_files_path)
with mp.Pool(processes=3) as pool:
    pool.map(partial_filter_events_features, args)