import csv

import pandas as pd
import progressbar
import os

import helper

def create_columns_dict(columns):
    dict_columns = dict()
    for column in columns:
        column_attribute = column.split('_')[:2]
        if column_attribute not in dict_columns.keys():
            dict_columns[column_attribute] = []
        dict_columns[column_attribute].append(column)
    return dict_columns


csv_file_paths = [
    'csvs/dataset_organism_resistance_manualRemove.csv',
    'csvs/dataset_organism_resistance_manualRemove_IG.csv',
    'csvs/dataset_organism_resistance_manualRemove_noUseless.csv',
    'csvs/dataset_organism_resistance_manualRemove_noUseless_wrapper.csv',
    'csvs/dataset_organism_resistance_noUseless.csv',
    'csvs/dataset_organism_resistance_noUseless_wrapper.csv',
    'csvs/dataset_organism_resistance.csv',
    'csvs/dataset_organism_resistance_IG.csv'
]
csv_columns = dict()
for csv_file in csv_file_paths:
    csv_file = csv_file.split('/')[1].split('.')[0]
    csv_columns[csv_file] = helper.get_attributes_from_csv(csv_file)

dataset_csv = pd.read_csv('dataset.csv')
events_files_path = './data_all/'
# Collect nominal data from values and create the binary representation


with progressbar.ProgressBar(max_value=len(dataset_csv)) as bar:
    i = 0
    events_columns = None
    for index, patient in dataset_csv.iterrows():
        print("===== {} =====".format(patient['icustay_id']))
        if os.path.exists(events_files_path + '{}.csv'.format(patient['icustay_id'])):
            # Get events and change nominal to binary
            events = pd.read_csv(events_files_path + '{}.csv'.format(patient['icustay_id']))
            if events_columns is None:
                events_columns = create_columns_dict(events.columns)
            for csv_file in csv_columns.keys():
                if not os.path.exists(csv_file):
                    os.mkdir(csv_file)
                new_events = events[csv_columns[csv_file]]
                new_events.to_csv(csv_file + '/{}.csv'.format(patient['icustay_id']), quoting=csv.QUOTE_NONNUMERIC,
                                  index=False)
        i+= 1
        bar.update(i)