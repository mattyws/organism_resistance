import pandas as pd
import json

from sklearn.model_selection import train_test_split
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

class DataGenerator(object):

    def __init__(self, dataset, files_path):
        self.dataset_generator = dataset.iterrows()
        self.files_path = files_path

    def __iter__(self):
        return self

    def __next__(self):
        try:
            index, row = next(self.dataset_generator)
            event = pd.read_csv(self.files_path + '{}.csv'.format(row['icustay_id']))
            event.loc[:, "icustay_id"] = [row['icustay_id'] for i in range(len(event))]
            return event
        except StopIteration as spe:
            raise StopIteration()


parametersFilePath = "parameters/data_parameters.json"

#Loading parameters file
print("========= Loading Parameters")
parameters = None
with open(parametersFilePath, 'r') as parametersFileHandler:
    parameters = json.load(parametersFileHandler)
if parameters is None:
    exit(1)


dataset = pd.read_csv(parameters['datasetCsvFilePath'])

all_events = pd.DataFrame([])
max_data = 70
y = dataset['class']

X_train, dataset, y_train, y_test = train_test_split(dataset, y, test_size=0.001, random_state=42, stratify=y)
y_test.index = dataset['icustay_id']

data_generator = DataGenerator(dataset, parameters['datasetFilesPath'])

print("Getting events")
for index, row in dataset.iterrows():
    event = pd.read_csv(parameters['datasetFilesPath']+'{}.csv'.format(row['icustay_id']))
    event.loc[:, "icustay_id"] = [row['icustay_id'] for i in range(len(event))]
    all_events = pd.concat([all_events, event], ignore_index=True)
all_events['event_timestamp'] = pd.to_datetime(all_events['event_timestamp'], format=parameters['datetimePattern'])
all_events = all_events.fillna(0)

extracted_features = extract_features(all_events, column_id='icustay_id', column_sort='event_timestamp')
impute(extracted_features)
features_filtered = select_features(extracted_features, y_test)
print(features_filtered)