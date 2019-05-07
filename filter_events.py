"""
Filter all events from chartevents and labevents for each patient at the dataset
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
    interpretation_label = "interpretation"
    ab_name_label = 'ANTIBODY'
    for event in events:
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


datetime_pattern = "%Y-%m-%d %H:%M:%S"
microbiologyevent_label = "microbiologyevents"
features_event_label = ['chartevents', 'labevents']
event_labels = ['CHARTEVENTS', 'LABEVENTS']
mimic_data_path = "/home/mattyws/Documentos/mimic/data/"

events_files_path = mimic_data_path + 'data_organism_resistence/'
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
patients.loc[:, 'ADMITTIME'] = pd.to_datetime(patients['ADMITTIME'], format=datetime_pattern)

json_files_path = mimic_data_path+"json/"
# files_paths = []
# for index, patient in patients.iterrows():
#     year = patient['suspected_infection_time_poe'].year
#     files_paths.append(json_files_path+str(year)+'/{}.json'.format(patient['hadm_id']))

# patients = open('sepsis_patients4')
# Loop through all patients that fits the sepsis 3 definition
dataset_csv = []
hadm_ids_added = []
ab_classes = get_antibiotics_classes()
for index, row in patients.iterrows():
    print("####### Admission {} #######".format(row['hadm_id']))
    year = row['suspected_infection_time_poe'].year
    json_path = json_files_path+str(year)+'/{}.json'.format(row['hadm_id'])
    if not os.path.exists(json_path):
        continue
    patient = json.load(open(json_path))
    patient_age = row['age']# get_patient_age(patient['subject_id'], patient['admittime'])
    intime = row['ADMITTIME'] # datetime.strptime(patient['admittime'], datetime_pattern)
    diff = row['suspected_infection_time_poe'] - intime
    if diff.days < 0 or (diff.days == 0 and diff.seconds/3600 < 12):
        continue
    events_in_patient = dict()
    window_start = row['suspected_infection_time_poe'] - timedelta(hours=12)
    cut_poe = row['suspected_infection_time_poe'] # intime + timedelta(hours=24)
    if microbiologyevent_label not in patient.keys() or (patient_age < 18 or patient_age > 80):
        print("Patient age: {}".format(patient_age))
        continue
    organism_resistance = get_organism_class(patient[microbiologyevent_label], ab_classes)
    print(window_start, cut_poe, row['intime'])
    for feature_event_label in features_event_label:
        print(len(patient[feature_event_label]))
        # Loading event csv
        if feature_event_label not in patient.keys():
            continue
        events_df = patient[feature_event_label]
        # Filter events that occurs between ICU intime and ICU outtime, as the csv corresponds to events that occurs
        # to all hospital admission
        print("==== Looping {} events for {} ====".format(feature_event_label, row['icustay_id']))
        for event in events_df:
            event['charttime'] = datetime.strptime(event['charttime'], datetime_pattern)
            if event['charttime'] >= window_start and event['charttime'] <= cut_poe:
                # Get values and store into a variable, just to read easy and if the labels change
                itemid = event['ITEMID']
                event_timestamp = event['charttime']
                event_value = event['valuenum']
                print(event_value)
                if event['valuenum'] is None:
                    print(itemid)
                    exit()
                # if feature_event_label == 'chartevents' and itemid in helper.FEATURES_ITEMS_TYPE.keys():
                #     if helper.FEATURES_ITEMS_TYPE[itemid] == helper.CATEGORICAL_LABEL:
                #         event_value = event['value']
                # If the id is not in events yet, create it and assign a empty dictionary to it
                if event_timestamp not in events_in_patient.keys():
                    events_in_patient[event_timestamp] = dict()
                    events_in_patient[event_timestamp]['event_timestamp'] = event_timestamp
                    # events_in_patient[itemid]['event_type'] = feature_event_label
                # If the timestamp from the event is in the event, assign the higher value between the tow of then
                # It is to check if a same event is masured more than one time at the same timestamp
                if itemid in events_in_patient[event_timestamp].keys():
                    if event_value > events_in_patient[event_timestamp][feature_event_label+'_'+itemid]:
                        events_in_patient[event_timestamp][feature_event_label+'_'+itemid] = event_value
                else:
                    # Else just create the field and assign its value
                    events_in_patient[event_timestamp][feature_event_label+'_'+itemid] = event_value
    print("Converting to dataframe")
    patient_data = pd.DataFrame([])
    for event_timestamp in events_in_patient.keys():
        events = pd.DataFrame(events_in_patient[event_timestamp], index=[0])
        patient_data = pd.concat([patient_data, events], ignore_index=True)
    if len(patient_data) != 0:
        print("==== Creating csv ====")
        patient_data.to_csv(events_files_path + '{}.csv'.format(row['icustay_id']), quoting=csv.QUOTE_NONNUMERIC,
                            index=False)
    else:
        print("Error in file {}, events is empty".format(patient['hadm_id']))
    if patient['hadm_id'] not in hadm_ids_added:
        dataset_csv.append(
            {'hadm_id': row['hadm_id'],
             'icustay_id' : row['icustay_id'],
             'intime': row['intime'],
             'admittime': row['ADMITTIME'],
             'class': organism_resistance,
             'sex' : row['gender'],
             'age' : row['age'],
             'ethnicity' : row['ethnicity']
             }
        )
        hadm_ids_added.append(row['icustay_id'])
    print(row['ADMITTIME'], row['intime'], row['suspected_infection_time_poe'])
    break
dataset_csv = pd.DataFrame(dataset_csv)
# print(len(dataset_csv))
dataset_csv.to_csv('dataset.csv')