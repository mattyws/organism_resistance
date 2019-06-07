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
print("Geral :", data["age"].mean())
print('Resistente', data_resistent_age["age"].mean())
print('Não resistente', data_nonresistent_age["age"].mean())
print("Distribuição genero")
print("Geral :", data["sex"].value_counts().to_dict())
print("Resistente", data_resistent['sex'].value_counts().to_dict())
print("Não resistente", data_nonresistent['sex'].value_counts().to_dict())
print("Etinicidade")
print("Resistente", data_resistent['ethnicity'].value_counts().to_dict())
print("Não resistente", data_nonresistent['ethnicity'].value_counts().to_dict())
print("Uso de vasopressores")
print("Resistente", data_resistent['vasopressor'].value_counts().to_dict())
print("Não resistente", data_nonresistent['vasopressor'].value_counts().to_dict())
print("Dias até a suspeita")
print("Geral :", data["days_until_suspicion"].mean())
print('Resistente', data_resistent_age["days_until_suspicion"].mean())
print('Não resistente', data_nonresistent_age["days_until_suspicion"].mean())

dict_patients = dict()
print("Loading patients")
patients_df = pandas.read_csv('PATIENTS.csv')
admissions_df = pandas.read_csv('ADMISSIONS.csv')
admissions_df.loc[:, "ADMITTIME"] = pandas.to_datetime(admissions_df['ADMITTIME'], format="%Y-%m-%d %H:%M:%S")
admissions_df.loc[:, "DISCHTIME"] = pandas.to_datetime(admissions_df['DISCHTIME'], format="%Y-%m-%d %H:%M:%S")
admissions_df.loc[:, "staytime"] = (admissions_df["DISCHTIME"] - admissions_df["ADMITTIME"]).apply(lambda x: x.days)
icustays_df = pandas.read_csv('ICUSTAYS.csv')
icustays_df.loc[:, "icustay_id"] = icustays_df['ICUSTAY_ID']
icustays_df.loc[:, "INTIME"] = pandas.to_datetime(icustays_df['INTIME'], format="%Y-%m-%d %H:%M:%S")
icustays_df.loc[:, "OUTTIME"] = pandas.to_datetime(icustays_df['OUTTIME'], format="%Y-%m-%d %H:%M:%S")
icustays_df.loc[:, "staytime"] = (icustays_df["OUTTIME"] - icustays_df["INTIME"]).apply(lambda x: x.days)
# admissions_df = pandas.merge(patients_df, admissions_df, how="inner", on=['SUBJECT_ID'])
admissions_df['hadm_id'] = admissions_df['HADM_ID']

mortality = pandas.merge(admissions_df, data, on=['hadm_id'], how="inner")
mortality = mortality.drop_duplicates(subset='hadm_id')

mortality_resistant = pandas.merge(admissions_df, data_resistent, on=['hadm_id'], how='inner')
mortality_resistant = mortality_resistant.drop_duplicates(subset='hadm_id')
mortality_nonresistant = pandas.merge(admissions_df, data_nonresistent, on=['hadm_id'], how='inner')
mortality_nonresistant = mortality_nonresistant.drop_duplicates(subset='hadm_id')
print("Mortalidade")
print("Geral", mortality['HOSPITAL_EXPIRE_FLAG'].value_counts().to_dict())
print("Resistente", mortality_resistant['HOSPITAL_EXPIRE_FLAG'].value_counts().to_dict())
print("Não resistente", mortality_nonresistant['HOSPITAL_EXPIRE_FLAG'].value_counts().to_dict())

print("Dias de estadia")
print("Hospital")
print("Geral", mortality['staytime'].mean())
print("Resistente", mortality_resistant['staytime'].mean())
print("Não resistente", mortality_nonresistant['staytime'].mean())

icustays = pandas.merge(icustays_df, data, on=['icustay_id'], how="inner")
icustays_resistant = pandas.merge(icustays_df, data_resistent, on=['icustay_id'], how="inner")
icustays_nonresistant = pandas.merge(icustays_df, data_nonresistent, on=['icustay_id'], how="inner")
print("ICU")
print("Geral", icustays['staytime'].mean())
print("Resistente", icustays_resistant['staytime'].mean())
print("Não resistente", icustays_nonresistant['staytime'].mean())