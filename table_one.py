import csv
import json

import pandas

def get_organism_class(events):
    org_item_label = "ORG_ITEMID"
    interpretation_label = "interpretation"
    organism_count = dict()
    for event in events:
        if org_item_label in event.keys():
            if event[org_item_label] not in organism_count.keys():
                organism_count[event[org_item_label]] = 0
            if event[interpretation_label] == 'R':
                organism_count[event[org_item_label]] += 1
                if organism_count[event[org_item_label]] == 3:
                    return "R"
    return "S"

data = pandas.read_csv('csvs/dataset_organism_resistance_generated.csv')
data_resistent = data[data['organism_resistence'] == 'R']
data_nonresistent = data[data['organism_resistence'] == 'S']

data_resistent_age = data_resistent[data_resistent["age"] >= 18]
data_resistent_age = data_resistent_age[data_resistent_age["age"] <= 80]

data_nonresistent_age = data_nonresistent[data_nonresistent["age"] >= 18]
data_nonresistent_age = data_nonresistent_age[data_nonresistent_age["age"] <= 80]

print("Quantidade de pacientes com bactérias resistentes", len(data_resistent))
print("Quantidade de pacientes com bactérias não resistentes", len(data_nonresistent))
print("Média de idade")
print('Resistente', data_resistent_age["age"].mean())
print('Não resistente', data_nonresistent_age["age"].mean())
print("Distribuição genero")
print("Resistente", data_resistent['sex'].value_counts().to_dict())
print("Não resistente", data_nonresistent['sex'].value_counts().to_dict())
print("Etinicidade")
print("Resistente", data_resistent['ethnicity'].value_counts().to_dict())
print("Não resistente", data_nonresistent['ethnicity'].value_counts().to_dict())
print("Uso de vasopressores")
print("Resistente", data_resistent['vasopressor'].value_counts().to_dict())
print("Não resistente", data_nonresistent['vasopressor'].value_counts().to_dict())

# print("Ventilação mecânica")
# print("Resistente", data_resistent['item_467'].value_counts().to_dict())
# print("Não resistente", data_nonresistent['item_467'].value_counts().to_dict())

dict_patients = dict()
print("Loading patients")
patients_df = pandas.read_csv('PATIENTS.csv')
admissions_df = pandas.read_csv('ADMISSIONS.csv')
# admissions_df = pandas.merge(patients_df, admissions_df, how="inner", on=['SUBJECT_ID'])
admissions_df['hadm_id'] = admissions_df['HADM_ID']

mortality_resistant = pandas.merge(admissions_df, data_resistent, on=['hadm_id'], how='inner')
mortality_resistant = mortality_resistant.drop_duplicates(subset='hadm_id')
mortality_nonresistant = pandas.merge(admissions_df, data_nonresistent, on=['hadm_id'], how='inner')
mortality_nonresistant = mortality_nonresistant.drop_duplicates(subset='hadm_id')

print(len(mortality_resistant))
print(len(mortality_nonresistant))

print("Resistente", mortality_resistant['HOSPITAL_EXPIRE_FLAG'].value_counts().to_dict())
print("Não resistente", mortality_nonresistant['HOSPITAL_EXPIRE_FLAG'].value_counts().to_dict())
