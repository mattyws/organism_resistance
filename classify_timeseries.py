import csv
import json
import os
import pickle
import pandas as pd
import numpy as np

import keras

from keras.callbacks import ModelCheckpoint
from sklearn.model_selection._split import StratifiedKFold

from helpers.data_generators import LongitudinalDataGenerator
from helpers.keras_callbacks import SaveModelEpoch
from helpers.model_creators import MultilayerKerasRecurrentNNCreator
from helpers.metrics import f1, precision, recall

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATETIME_PATTERN = "%Y-%m-%d %H:%M:%S"

parametersFilePath = "parameters/classify_chartevents_parameters.json"

#Loading parameters file
print("========= Loading Parameters")
parameters = None
with open(parametersFilePath, 'r') as parametersFileHandler:
    parameters = json.load(parametersFileHandler)
if parameters is None:
    exit(1)

if not os.path.exists(parameters['modelCheckpointPath']):
    os.mkdir(parameters['modelCheckpointPath'])

data_csv = pd.read_csv('dataset.csv')
data_csv = data_csv.sort_values(['hadm_id'])
data = np.array([itemid for itemid in list(data_csv['hadm_id'])
                 if os.path.exists(parameters['dataPath'] + '{}.csv'.format(itemid))])
data_csv = data_csv[data_csv['hadm_id'].isin(data)]
data = np.array([parameters['dataPath'] + '{}.csv'.format(itemid) for itemid in data])
classes = np.array([ [0] if c == 'S' else [1] for c in list(data_csv['class'])])
classes_for_stratified = np.array([ 0 if c == 'S' else 1 for c in list(data_csv['class'])])
print('S', len([c for c in classes if c == [0]]))
print('R', len([c for c in classes if c == [1]]))
# Using a seed always will get the same data split even if the training stops
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)

# Get input shape
aux = pd.read_csv(data[0])
inputShape = (parameters['dataLength'], len(aux.columns)-1)

config = None
if os.path.exists(parameters['modelConfigPath']):
    with open(parameters['modelConfigPath'], 'r') as configHandler:
        config = json.load(configHandler)

i = 0
# ====================== Script that start training new models
with open(parameters['resultFilePath'], 'a+') as cvsFileHandler: # where the results for each fold are appended
    dictWriter = None
    for trainIndex, testIndex in kf.split(data, classes_for_stratified):
        if config is not None and config['fold'] > i:
            print("Pass fold {}".format(i))
            i += 1
            continue
        if config is not None and config['epoch'] == parameters['trainingEpochs']:
            print("Pass fold {}-".format(i))
            i += 1
            continue
        print("======== Fold {} ========".format(i))

        # If exists a valid config  to resume a training
        if config is not None and config['fold'] == i and config['epoch'] < parameters['trainingEpochs']:
            epochs = parameters['trainingEpochs'] - config['epoch']

            print("========= Loading generators")
            with open(parameters['trainingGeneratorPath'], 'rb') as trainingGeneratorHandler:
                dataTrainGenerator = pickle.load(trainingGeneratorHandler)

            with open(parameters['testingGeneratorPath'], 'rb') as testingGeneratorHandler:
                dataTestGenerator = pickle.load(testingGeneratorHandler)

            kerasAdapter = MultilayerKerasRecurrentNNCreator.create_from_path(config['filepath'],
                                                    custom_objects={'f1':f1, 'precision':precision, 'recall':recall})
            configSaver = SaveModelEpoch(parameters['modelConfigPath'],
                                         parameters['modelCheckpointPath'] + 'fold_' + str(i), i, alreadyTrainedEpochs=config['epoch'])
        else:
            dataTrainGenerator = LongitudinalDataGenerator(data[trainIndex], classes[trainIndex], parameters['batchSize'])
            dataTestGenerator = LongitudinalDataGenerator(data[testIndex], classes[testIndex], parameters['batchSize'])
            print("========= Saving generators")
            with open(parameters['trainingGeneratorPath'], 'wb') as trainingGeneratorHandler:
                pickle.dump(dataTrainGenerator, trainingGeneratorHandler, pickle.HIGHEST_PROTOCOL)

            with open(parameters['testingGeneratorPath'], 'wb') as testingGeneratorHandler:
                pickle.dump(dataTestGenerator, testingGeneratorHandler, pickle.HIGHEST_PROTOCOL)

            modelCreator = MultilayerKerasRecurrentNNCreator(inputShape, parameters['outputUnits'], parameters['numOutputNeurons'],
                                                             loss=parameters['loss'], layersActivations=parameters['layersActivations'],
                                                             gru=parameters['gru'], use_dropout=parameters['useDropout'],
                                                             dropout=parameters['dropout'],
                                                             metrics=[f1, precision, recall, keras.metrics.binary_accuracy])
            kerasAdapter = modelCreator.create()
            epochs = parameters['trainingEpochs']
            configSaver = SaveModelEpoch(parameters['modelConfigPath'],
                                         parameters['modelCheckpointPath'] + 'fold_' + str(i), i)

        modelCheckpoint = ModelCheckpoint(parameters['modelCheckpointPath']+'fold_'+str(i))
        kerasAdapter.fit(dataTrainGenerator, epochs=epochs, batch_size=len(dataTrainGenerator),
                         validationDataGenerator=dataTestGenerator, validationSteps=len(dataTestGenerator),
                         callbacks=[modelCheckpoint, configSaver])
        result = kerasAdapter.evaluate(dataTestGenerator, batch_size=len(dataTestGenerator))
        result["fold"] = i
        if dictWriter is None:
            dictWriter = csv.DictWriter(cvsFileHandler, result.keys())
        if result['fold'] == 0:
            dictWriter.writeheader()
        dictWriter.writerow(result)
        i += 1
