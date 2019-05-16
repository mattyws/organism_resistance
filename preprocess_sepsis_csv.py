import csv
import re
from collections import Counter

import numpy
import pandas as pd


# table = pd.read_csv('sepsis_file3.csv')
# table = table.filter(regex="lab_")
# for column in table.columns:
#     print(column, table[column].dtype, table[column].unique())
# exit()


range_re = re.compile('\d+-\d+')
minutes_re = re.compile('\d+\"|\d+ min|\d+min|q\d+min')
print("Get row types")
with open('dataset_organism_resistance.csv', 'r') as sepsisHandler:
    dictReader = csv.DictReader(sepsisHandler, quoting=csv.QUOTE_NONNUMERIC)
    new_data = []
    dictTypes = dict()
    for key in dictReader.fieldnames:
        dictTypes[key] = list()
    try:
        for row in dictReader:
            for key in row.keys():
                if row[key] != "?" and row[key] is not None and key is not None:
                    try:
                        typeof = type(float(row[key]))
                    except:
                        typeof = type(row[key])
                    dictTypes[key].append(typeof)
    except:
        print(dictReader.fieldnames)
    for key in dictTypes.keys():
        dictTypes[key] = Counter(dictTypes[key])
        # print(key, dictTypes[key], len(dictTypes[key]))
# exit()

print("Preprocess")
events_not_known = set()
with open('dataset_organism_resistance.csv', 'r') as sepsisHandler:
    dictReader = csv.DictReader(sepsisHandler, quoting=csv.QUOTE_NONNUMERIC)
    for row in dictReader:
        print("#### {}".format(row['icustay_id']))
        empty_key = None
        for key in row.keys():
            if len(dictTypes[key]) < 2:
                continue
            try:
                row[key] = float(row[key])
            except:
                row[key] = str(row[key])
            # if isinstance(row[key], str) and range_re.match(row[key]):
            #     numbers = re.findall('\d+', row[key])
            #     numbers = [float(n) for n in numbers]
            #     row[key] = sum(numbers) / len(numbers) if len(numbers) > 0 else "?"
            #     if type(float) not in dictTypes[key]:
            #         dictTypes[key].append(type(float))
            if isinstance(row[key], str) and dictTypes[key][float] > dictTypes[key][str] and row[key] != "?":
                if range_re.match(row[key])	:
                    numbers = re.findall('\d+',row[key])
                    numbers = [float(n) for n in numbers]
                    row[key] = sum(numbers) / len(numbers) if len(numbers) > 0 else "?"
                elif row[key].startswith('LESS THAN') or row[key].startswith('<') or \
                        row[key].startswith('LESS THEN'):
                    numbers = re.findall('\d+',row[key])
                    row[key] = float(numbers[0]) if len(numbers) > 0 else "?"
                elif row[key].startswith('GREATER THAN') or row[key].startswith('>') or \
                        row[key].startswith('GREATER THEN'):
                    numbers = re.findall('\d+',row[key])
                    row[key] = float(numbers[0]) if len(numbers) > 0 else "?"
                elif 'HEMOLYSIS FALSELY INCREASES THIS RESULT' == row[key]:
                    row[key] = "?"
                elif row[key] == 'NEG':
                    row[key] = '-1'
                elif row[key] == 'NEG':
                    row[key] = '-1'
                elif 'ERROR' in row[key]:
                    row[key] = "?"
                elif row[key].lower() == 'tr':
                    row[key] = "?"
                elif row[key].lower() == 'n':
                    row[key] = "?"
                elif row[key] == numpy.nan:
                    row[key] = "?"
                elif row[key] == 'NONE' or row[key].lower() == 'notdone':
                    row[key] = "?"
                elif row[key] == '-' or row[key] == '.':
                    row[key] = "?"
                elif row[key].lower() == 'tntc':
                    row[key] = "?"
                elif minutes_re.match(row[key].lower()):
                    numbers = re.findall('\d+', row[key])
                    row[key] = float(numbers[0]) if len(numbers) > 0 else "?"
                else:
                    # print("=======================++", row[key], dictTypes[key])
                    events_not_known.add(row[key])
                    row[key] = "?"
            elif isinstance(row[key], str) and dictTypes[key][float] < dictTypes[key][str] and row[key] != "?":
                row[key] = "?"
        row.pop(None, None)
        new_data.append(row)
print(events_not_known)
print("Converting to pandas dataframe")
new_data = pd.DataFrame(new_data)
if 'Unnamed: 0' in new_data.columns:
    new_data = new_data.drop(columns=['Unnamed: 0'])
if '' in new_data.columns:
    new_data = new_data.drop(columns=[''])
columns_to_remove = []
for column in new_data.columns:
    if len(new_data[column].unique()) == 1:
        if new_data[column].unique()[0] == "?":
            print(column)
            columns_to_remove.append(column)
new_data = new_data.drop(columns=columns_to_remove)
print("Creating file")
new_data.to_csv("dataset_organism_resistance_preproc.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
# with open('dataset_organism_resistance_2.csv', 'w') as newFileHandler:
#     header = list(new_data[0].keys())
#     header.sort()
#     dictWriter = csv.DictWriter(newFileHandler, header, quoting=csv.QUOTE_NONNUMERIC)
#     dictWriter.writeheader()
#     for row in new_data:
#         dictWriter.writerow(row)
