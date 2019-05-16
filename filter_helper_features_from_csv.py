import csv

import pandas as pd
import helper

data = pd.read_csv('dataset_organism_resistance.csv')
columns_to_remove = []

for column in data.columns:
    if column.startswith('labitems_'):
        if column.split('_')[1] not in helper.FEATURES_LABITEMS_LABELS.keys():
            columns_to_remove.append(column)
    if column.startswith('chartevents_'):
        if column.split('_')[1] not in helper.FEATURES_ITEMS_LABELS.keys():
            columns_to_remove.append(column)

data = data.drop(columns=columns_to_remove)
data.to_csv('dataset_organism_resistance_manual.csv', quotechar="\"", quoting=csv.QUOTE_NONNUMERIC)