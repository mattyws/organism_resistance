import json

import pandas as pd

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
    for key in organism_count:
        if len(organism_count[key]) >= 3:
            return organism_count, "R"
    return organism_count, "S"

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

datetime_pattern = "%Y-%m-%d %H:%M:%S"

dataset = pd.read_csv('dataset.csv')
dataset.loc[:, 'admittime'] = pd.to_datetime(dataset['admittime'], format=datetime_pattern)
d_items = pd.read_csv('D_ITEMS.csv')
ab_classes = get_antibiotics_classes()

mimic_data_path = "/home/mattyws/Documents/mimic_data/"
json_files_path = mimic_data_path+"json/"

organisms = pd.DataFrame([])
for index, row in dataset.iterrows():
    print("####### Icustay {} #######".format(row['icustay_id']))
    year = row['admittime'].year
    json_path = json_files_path + str(year) + '/{}.json'.format(row['hadm_id'])
    patient = json.load(open(json_path))
    organism_count, organism_resistance = get_organism_class(patient['microbiologyevents'], ab_classes)
    organism_labels = d_items[d_items['ITEMID'].isin(organism_count.keys())]
    organism_count_label = dict()
    organism_count_label['icustay_id'] = row['icustay_id']
    for key in organism_count.keys():
        key_label = organism_labels[organism_labels['ITEMID'] == int(key)].iloc[0]['LABEL']
        organism_count_label[key_label+'_{}'.format(key)] = len(organism_count[key])
        if len(organism_count[key]) >= 3:
            organism_count_label['org_name'] = key_label
    organism_count_label = pd.DataFrame(organism_count_label, index=[0])
    organisms = pd.concat([organisms, organism_count_label], ignore_index=True)

dataset = pd.merge(dataset, organisms, how='inner', on=['icustay_id'])
dataset.to_csv('dataset_organisms_labels.csv')