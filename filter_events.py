"""
Filter all events from chartevents and labevents for each patient that have a suspicion of infection, using the work
"A Comparative Analysis of Sepsis Identification Methods in an Electronic Database".
Follow the repository at github (https://github.com/alistairewj/sepsis3-mimic) for more details.
"""
import csv
import json
import math
import os
import pprint
import re
import time

import pandas as pd
from datetime import datetime, timedelta

import helper

def get_organism_class(events, ab_classes):
    organism_count = dict()
    org_item_label = "ORG_ITEMID"
    interpretation_label = "INTERPRETATION"
    ab_name_label = 'AB_NAME'
    for index, event in events.iterrows():
        if org_item_label in event.keys():
            if event[org_item_label] not in organism_count.keys():
                organism_count[event[org_item_label]] = set()
            if event[interpretation_label] == 'R':
                organism_count[event[org_item_label]].add(ab_classes[event[ab_name_label]])
                if len(organism_count[event[org_item_label]]) == 3:
                    return "R"
    return "S"

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

def get_patient_age(patient_id, admittime_str):
    admittime = time.strptime(admittime_str, datetime_pattern)
    with open('PATIENTS.csv', 'r') as patient_file_handler:
        dict_reader = csv.DictReader(patient_file_handler)
        for row in dict_reader:
            if row['subject_id'.upper()] == patient_id:
                dob = time.strptime(row['DOB'], datetime_pattern)
                difference = admittime.tm_year - dob.tm_year - ((admittime.tm_mon, dob.tm_mday) < (admittime.tm_mon, dob.tm_mday))
                return difference
    return None

parametersFilePath = "parameters/data_parameters.json"

#Loading parameters file
print("========= Loading Parameters")
parameters = None
with open(parametersFilePath, 'r') as parametersFileHandler:
    parameters = json.load(parametersFileHandler)
if parameters is None:
    exit(1)

# Defining some helper variables
datetime_pattern = parameters['datetimePattern']
features_event_label = ['chartevents', 'labevents']
mimic_data_path = parameters['mimicDataPath']
microbiologyevents_path = mimic_data_path + 'MICROBIOLOGYEVENTS/'
events_directories = [mimic_data_path + 'CHARTEVENTS/CHARTEVENTS_', mimic_data_path+'LABEVENTS/LABEVENTS_']

events_files_path = parameters['datasetFilesPath']
if not os.path.exists(events_files_path):
    os.mkdir(events_files_path)

patients = pd.read_csv('sepsis3-df-no-exclusions.csv')
patients = patients[patients["suspicion_poe"]]
patients.loc[:, 'suspected_infection_time_poe'] = pd.to_datetime(patients['suspected_infection_time_poe'],
                                                                 format=datetime_pattern)
patients.loc[:, 'intime'] = pd.to_datetime(patients['intime'], format=datetime_pattern)

admissions = pd.read_csv('ADMISSIONS.csv')
admissions['hadm_id'] = admissions['HADM_ID']

patients = pd.merge(patients, admissions, how="inner", on=['hadm_id'])
patients.loc[:, 'admittime'] = pd.to_datetime(patients['ADMITTIME'], format=datetime_pattern)

dataset_csv = []
hadm_ids_added = []
ab_classes = get_antibiotics_classes()
for index, row in patients.iterrows():
    print("####### Icustay {} Admission {} #######".format(row['icustay_id'], row['hadm_id']))
    year = row['admittime'].year
    patient_age = row['age']# get_patient_age(patient['subject_id'], patient['admittime'])
    if not os.path.exists(microbiologyevents_path+'MICROBIOLOGYEVENTS_{}.csv'.format(row['hadm_id'])) :
        print("{} does not exist!".format(microbiologyevents_path+'{}.csv'.format(row['hadm_id'])))
        continue
    if patient_age < 18 or patient_age > 80:
        print("Patient age: {}".format(patient_age))
        continue
    intime = row['intime'] # datetime.strptime(patient['admittime'], datetime_pattern)
    diff = intime - row['suspected_infection_time_poe']
    if diff.days < 0:
        continue
    if diff.days == 0 and diff.seconds/3600 < 12:
        window_start = row['intime']
        cut_poe = row['intime'] + timedelta(hours=12)  # intime + timedelta(hours=24)
    else:
        window_start = row['suspected_infection_time_poe'] - timedelta(hours=12)
        cut_poe = row['suspected_infection_time_poe']  # intime + timedelta(hours=24)
    events_in_patient = dict()
    for feature_event_label, events_directory in zip(features_event_label, events_directories):
        # Loading event csv
        if not os.path.exists(events_directory+'{}.csv'.format(row['hadm_id'])):
            continue
        events_df = pd.read_csv(events_directory+'{}.csv'.format(row['hadm_id']))
        # Filter events that occurs between ICU intime and ICU outtime, as the csv corresponds to events that occurs
        # to all hospital admission
        print("==== Looping {} events for {} ====".format(feature_event_label, row['icustay_id']))
        for index2, event in events_df.iterrows():
            event['CHARTTIME'] = datetime.strptime(event['CHARTTIME'], datetime_pattern)
            if event['CHARTTIME'] >= window_start and event['CHARTTIME'] <= cut_poe:
                # Get values and store into a variable, just to read easy and if the labels change
                itemid = event['ITEMID']
                event_timestamp = event['CHARTTIME']
                if event['VALUENUM'] is None:
                    event_value = str(event['VALUE'])
                else:
                    event_value = float(event['VALUENUM'])
                # If the id is not in events yet, create it and assign a empty dictionary to it
                if event_timestamp not in events_in_patient.keys():
                    events_in_patient[event_timestamp] = dict()
                    events_in_patient[event_timestamp]['event_timestamp'] = event_timestamp
                    # events_in_patient[itemid]['event_type'] = feature_event_label
                # If the timestamp from the event is in the event, assign the higher value between the tow of then
                # It is to check if a same event is masured more than one time at the same timestamp
                if itemid in events_in_patient[event_timestamp].keys():
                    if event_value > events_in_patient[event_timestamp][feature_event_label+'_'+itemid]:
                        events_in_patient[event_timestamp][feature_event_label+'_'+str(itemid)] = event_value
                else:
                    # Else just create the field and assign its value
                    events_in_patient[event_timestamp][feature_event_label+'_'+str(itemid)] = event_value
    print("Converting to dataframe")
    patient_data = pd.DataFrame([])
    for event_timestamp in events_in_patient.keys():
        events = pd.DataFrame(events_in_patient[event_timestamp], index=[0])
        patient_data = pd.concat([patient_data, events], ignore_index=True)
    if len(patient_data) != 0:
        print("==== Creating csv ====")
        patient_data = patient_data.sort_values(by=['event_timestamp'])
        patient_data.to_csv(events_files_path + '{}.csv'.format(row['icustay_id']), quoting=csv.QUOTE_NONNUMERIC,
                            index=False)
        patient_microbiologyevents = pd.read_csv(microbiologyevents_path+'MICROBIOLOGYEVENTS_{}.csv'.format(row['hadm_id']))
        organism_resistance = get_organism_class(patient_microbiologyevents, ab_classes)
        if row['icustay_id'] not in hadm_ids_added:
            dataset_csv.append(
                {'hadm_id': row['hadm_id'],
                 'icustay_id': row['icustay_id'],
                 'intime': row['intime'],
                 'admittime': row['ADMITTIME'],
                 'class': organism_resistance,
                 'sex': row['gender'],
                 'age': row['age'],
                 'ethnicity': row['ethnicity'],
                 'window_starttime': window_start,
                 'window_endtime': cut_poe
                 }
            )
            hadm_ids_added.append(row['icustay_id'])
    else:
        print("Error in file {}, events is empty".format(row['icustay_id']))
dataset_csv = pd.DataFrame(dataset_csv)
# print(len(dataset_csv))
dataset_csv.to_csv('dataset.csv')