import json
import os
import re
import time
import pprint
import statistics
from collections import Counter
import sys
import csv
from datetime import datetime

import helper
import pandas as pd
import numpy as np

pp = pprint.PrettyPrinter(indent=5)
date_pattern = "%Y-%m-%d"
datetime_pattern = "%Y-%m-%d %H:%M:%S"
itemid_label = 'ITEMID'
valuenum_label = 'valuenum'
value_label = 'value'
labitems_prefix = 'labevents_'
items_prefix = 'chartevents_'
mean_key = 'mean'
std_key = 'std'
csv_file_name = "organism_resistance_dataset_2.csv"
class_label = "organism_resistence"
interpretation_label = "interpretation"
org_item_label = "ORG_ITEMID"
ab_name_label = 'ANTIBODY'
microbiologyevent_label = "microbiologyevents"
patient_file = 'PATIENTS.csv'
sofa_file = 'sofa.csv'
vasopressor_file = 'vasopressor_durations.csv'

gender_label = 'sex'
ethnicity_label = 'ethnicity'
age_label = 'age'
sofa_label = 'sofa'
birth_label = 'DOB'
vaso_label = 'vasopressor'
charttime_label = 'charttime'
itemid_label = 'ITEMID'
item_label = 'ITEM'

def transform_equal_columns(row):
    for prefix in [items_prefix, labitems_prefix]:
        pairs = []
        if prefix == labitems_prefix:
            pairs = helper.ARE_EQUAL_LAB
        elif prefix == items_prefix:
            pairs = helper.ARE_EQUAL_CHART
        for pair in pairs:
            first_in_keys = prefix+pair[0] in row.keys()
            second_in_keys = prefix+pair[1] in row.keys()
            if first_in_keys and second_in_keys:
                row[prefix + pair[0]].extend(row[prefix + pair[1]])
                row.pop(prefix + pair[1])
            elif second_in_keys and not first_in_keys:
                row[prefix + pair[0]] = row[prefix + pair[1]]
                row.pop(prefix + pair[1])
    return row


def farenheit_to_celcius(events):
    for key in events.keys():
        if key in helper.FARENHEIT_ID:
            events[key] = [helper.CELCIUS(temp) for temp in events[key]]
    return events


def change_o2_delivery_device_values(events):
    if 'chartevents_467' in events.keys():
        new_value = []
        for value in events['chartevents_467']:
            if value == 'Endotracheal tube' or value == 'Tracheostomy tube':
                new_value.append('Endotracheal/Tracheostomy tube')
            elif value is not None or len(value) != 0:
                new_value.append('Other')
            else:
                new_value.append(None)
        events['chartevents_467'] = new_value
    return events


def transform_all_features_to_row(events):
    range_re = re.compile('\d+-\d+')
    number_plus = re.compile('\d+\+')
    events = farenheit_to_celcius(events)
    row = transform_equal_columns(events)
    row = change_o2_delivery_device_values(row)
    # Removing NaN
    for key in row.keys():
        row[key] = [x for x in row[key] if str(x) != 'nan']
    # Loop all keys in the dictionary
    for key in row.keys():
        # This will register the type of each value in the series
        types = set()
        for value in row[key]:
            try:
                value = float(value)
            except:
                value = str(value)
            types.add(type(value))
        # Change to list to handle better
        types = list(types)
        # If the list has only one type in
        if len(types) == 1:
            # If they are numeric, get the mean
            if types[0] == type(int) or types[0] == type(float):
                row[key] = sum(row[key]) / len(row[key])
            else:
                # If they are string, get the most common
                row[key] = Counter(row[key]).most_common(1)[0][0]
        else:
            # Here we have mixed types on the series, here we will handle the most known cases
            # It is assumed that the final value are numerics
            for i in range(len(row[key])):
                try:
                    row[key][i] = float(row[key][i])
                except:
                    row[key][i] = str(row[key][i])
                if isinstance(row[key][i], str):
                    if row[key][i].lower() == 'notdone':
                        row[key][i] = 0
                    elif row[key][i].lower() == 'neg':
                        row[key][i] = -1
                    elif row[key][i].lower() == 'tr':
                        row[key][i] = None
                    elif row[key][i] == '-':
                        row[key][i] = 0
                    elif len(row[key][i].strip() ) == 0:
                        row[key][i] = None
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
                    elif row[key][i] == 'HIGH':
                        row[key][i] = 0
                    elif row[key][i] == 'no data':
                        row[key][i] = 0
                    elif 'UNABLE TO REPORT' in row[key][i] or 'VERIFIED BY REPLICATE ANALYSIS' in row[key][i]:
                        row[key][i] = None
                    elif 'ERROR' in row[key][i] or 'UNABLE' in row[key][i]:
                        row[key][i] = None
                    elif 'VERIFIED BY DILUTION' in row[key][i]:
                        row[key][i] = None
                    elif row[key][i] == 'FEW':
                        row[key][i] = 1
                    elif row[key][i] == 'MOD':
                        row[key][i] = 2
                    elif row[key][i] == 'MANY':
                        row[key][i] = 3
                    else:
                        print(row[key][i], "===============================")
                        row[key][i] = None
            row[key] = [w for w in row[key] if w is not None]
            if len(row[key]) > 0:
                try:
                    row[key] = sum(row[key]) / len(row[key])
                except:
                    print("Deu erro aqui: ", key, row[key], '====================================')
                    row[key] = row[key][0]
                    continue
            else:
                row[key] = None
    try:
        row = pd.DataFrame(row, index=[0])
    except:
        pp.pprint(row)
    return row

def get_antibiotics_classes():
    antibiotics_classes = dict()
    with open('AB_class') as antibiotics_classes_handler:
        antibiotics = []
        ab_class = ''
        for line in antibiotics_classes_handler:
            if len(line.strip()) != 0:
                if line.startswith('\t'):
                    antibiotics.append(line.strip())
                else:
                    if len(antibiotics) != 0:
                        for antibiotic in antibiotics:
                            antibiotics_classes[antibiotic] = ab_class
                    ab_class = line.strip()
                    antibiotics = []
        if len(antibiotics) != 0:
            for antibiotic in antibiotics:
                antibiotics_classes[antibiotic] = ab_class
    return antibiotics_classes

def get_admission_vasopressor(icustay_id, vasopressor_durations):
    return icustay_id in vasopressor_durations['icustay_id'].values

mimic_data_path = "/home/mattyws/Documents/mimic_data/"
events_files_path = mimic_data_path + 'data_organism_resistence/'

dataset_patients = pd.read_csv('dataset.csv')
dataset_patients.loc[:, 'intime'] = pd.to_datetime(dataset_patients['intime'], format=datetime_pattern)

sepsis3_df = pd.read_csv('sepsis3-df-no-exclusions.csv')
sepsis3_df.loc[:, 'intime'] = pd.to_datetime(sepsis3_df['intime'], format=datetime_pattern)

all_size = 0
filtered_objects_total_size = 0
table = pd.DataFrame([])
not_processes_files = 0
patients_with_pressure = 0
total_events_measured = 0
total_labevents_measured = 0
labitems_dict = dict()
chartevents_dict = dict()
ab_classes = get_antibiotics_classes()
vasopressor_durations = pd.read_csv(vasopressor_file)
for index, patient in dataset_patients.iterrows():
    print("#### Admission {} Icustay {}".format(patient['hadm_id'], patient['icustay_id']))

    num_previous_admission = len(sepsis3_df[ (sepsis3_df['hadm_id'] == patient['hadm_id']) &
                                                   (sepsis3_df['intime'] < patient['intime'])
                                 ])
    num_previous_infected_admission = len(sepsis3_df[(dataset_patients['hadm_id'] == patient['hadm_id']) &
                                            (sepsis3_df['intime'] < patient['intime']) & (sepsis3_df['suspicion_poe'])
                                            ])
    sepsis3_patient = sepsis3_df[sepsis3_df['icustay_id'] == patient['icustay_id']].iloc[0]
    suspected_infection = datetime.strptime(sepsis3_patient['suspected_infection_time_poe'], datetime_pattern)
    diff_days = (suspected_infection - patient['intime']).days
    patient_csv = pd.read_csv(events_files_path + '{}.csv'.format(patient['icustay_id'])).to_dict('list')

    # patient_dict = dict()


    row_object = transform_all_features_to_row(patient_csv)
    # row_labevent = transform_all_features_to_row(filtered_labevents_object, prefix=labitems_prefix)

    row_object['hadm_id'] = patient['hadm_id']
    row_object['icustay_id'] = patient['icustay_id']
    row_object[gender_label] = patient[gender_label]
    row_object[ethnicity_label] = patient[ethnicity_label]
    row_object[age_label.lower()] = patient['age']
    row_object['prev_hadm'] = num_previous_admission
    row_object['prev_infection_hadm'] = num_previous_infected_admission
    row_object['days_until_suspicion'] = diff_days
    # row_object[sofa_label] = get_admission_sofa(patient['hadm_id'])
    row_object[vaso_label] = get_admission_vasopressor(patient['icustay_id'], vasopressor_durations)
    row_object[class_label] = patient['class']
    table = pd.concat([table, row_object], ignore_index=True)

table.to_csv(csv_file_name, na_rep="?", quoting=csv.QUOTE_NONNUMERIC, index=False)

print("Number of files that do not had microbiologyevents : {}".format(not_processes_files))
print("Size of files processed : {} bytes".format(all_size))
print("Total size of filtered variables : {} bytes".format(filtered_objects_total_size))
print("Total events measured: {} chartevents, {} labevents".format(total_events_measured, total_labevents_measured))

