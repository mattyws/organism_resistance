"""
Run a test on all classifiers to generate new results.
It does it respecting the fold on which each classifier was trained
"""

import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics.classification import f1_score, precision_score, recall_score, cohen_kappa_score, accuracy_score, \
    confusion_matrix
from sklearn.metrics.ranking import roc_auc_score
from sklearn.model_selection._split import StratifiedKFold


def preprocess(data):
    data = pd.get_dummies(data, dummy_na=False)
    data = data.fillna(0)
    # for column in data:
    #     if data.dtypes[column] == float:
    #         data[column].fillna(0, inplace=True)
    #     elif data.dtypes[column] == int:
    #         data[column].fillna(0, inplace=True)
    return data

def normalize(df, mean, std):
    try:
        normalized_df = (df - mean) / std
    except Exception as e :
        print(e)
        print(list(df.columns))
        print(df.index.duplicated())
        # for column in df.columns:
        exit()
    normalized_df = normalized_df.replace([np.inf, -np.inf], np.nan)
    normalized_df.fillna(0, inplace=True)
    return normalized_df.values

def preprocess_classes(classes):
    return np.array([0 if c == 'S' else 1 for c in classes])

results_file_paths = [
    'results/result_dataset_organism_resistance_manualRemove.csv',
    'results/result_dataset_organism_resistance_manualRemove_IG.csv',
    'results/result_dataset_organism_resistance_manualRemove_noUseless.csv',
    'results/result_dataset_organism_resistance_manualRemove_noUseless_wrapper.csv',
    'results/result_dataset_organism_resistance_noUseless.csv',
    'results/result_dataset_organism_resistance_noUseless_wrapper.csv',
    'results/result_dataset_organism_resistance.csv',
    'results/result_dataset_organism_resistance_IG.csv',
    'results/result_dataset_organism_resistance_manual.csv'
]
class_label = "organism_resistence"

for result_file in results_file_paths:
    print("========== {} ==========".format(result_file))
    results_df = pd.read_csv(result_file)
    data = pd.read_csv('csvs/'+results_df.loc[0]['fname'])
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    classes = data[class_label]
    data = data.drop(columns=[class_label])
    data = preprocess(data)
    classes = preprocess_classes(classes)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=15)
    results = []
    folds = 0
    for train_index, test_index in kf.split(data, classes):
        print("====== Fold {} =====".format(folds))
        folds_classifiers = results_df[results_df['fold'] == folds]
        data_train, data_test = data.iloc[train_index], data.iloc[test_index]
        mean = data_train.mean()
        std = data_train.std()
        data_test = normalize(data_test, mean, std)
        classes_test = classes[test_index]
        for index, classifier_row in folds_classifiers.iterrows():
            print("====== {} =====".format(classifier_row['classifier']))
            classifier_fname = 'classifiers/{}_{}_fold{}.pkl'.format(classifier_row['fname'].split('.')[0],
                                                                     classifier_row['classifier'], classifier_row['fold'])
            classifier = joblib.load(open(classifier_fname, 'rb'))
            try:
                predicted = classifier.predict(data_test)
                metrics = dict()
                metrics['fscore'] = f1_score(classes_test, predicted, average='weighted')
                metrics['precision'] = precision_score(classes_test, predicted, average='weighted')
                metrics['recall'] = recall_score(classes_test, predicted, average='weighted')
                metrics['auc'] = roc_auc_score(classes_test, predicted, average='weighted')

                metrics['micro_f'] = f1_score(classes_test, predicted, average='weighted')
                metrics['micro_p'] = precision_score(classes_test, predicted, average='weighted')
                metrics['micro_r'] = recall_score(classes_test, predicted, average='weighted')

                metrics['fscore_b'] = f1_score(classes_test, predicted)
                metrics['precision_b'] = precision_score(classes_test, predicted)
                metrics['recall_b'] = recall_score(classes_test, predicted)

                metrics['kappa'] = cohen_kappa_score(classes_test, predicted)
                metrics['accuracy'] = accuracy_score(classes_test, predicted)
                tn, fp, fn, metrics['tp_rate'] = confusion_matrix(classes_test, predicted).ravel()
                metrics['tp_rate'] = metrics['tp_rate'] / (metrics['tp_rate'] + fn)
                metrics['classifier'] = classifier_row['classifier']
                metrics['fold'] = classifier_row['fold']
                metrics['fname'] = classifier_row['fname']
                results.append(metrics)
            except Exception as e:
                print(e)
                exit()
        folds += 1
    results = pd.DataFrame(results)
    results.to_csv(result_file.replace('results', 'results2'))