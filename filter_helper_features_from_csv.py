import csv

import pandas as pd
import helper

data = pd.read_csv('sepsis_file3.csv')
columns_to_remove = []

for column in data.columns:
    if column.startswith('lab_'):
        if column.split('_')[1] not in helper.FEATURES_LABITEMS_LABELS.keys():
            columns_to_remove.append(column)
    if column.startswith('item_'):
        if column.split('_')[1] not in helper.FEATURES_ITEMS_LABELS.keys():
            columns_to_remove.append(column)

data = data.drop(columns=columns_to_remove)
data.to_csv('filtered_sepsis_file3.csv', quotechar="\"", quoting=csv.QUOTE_NONNUMERIC)