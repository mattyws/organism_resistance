import csv
import itertools
import time

import numpy
import pandas
from sklearn import svm
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score, \
    confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np

import helper

def write_on_log(file, text):
    print(text)
    with open("log_{}".format(file), 'a+') as result_file_handler:
        result_file_handler.write(text+'\n')

def preprocess(data):
    data = pandas.get_dummies(data, dummy_na=False)
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

csv_file_paths = [
    'csvs/dataset_organism_resistance_manualRemove.csv',
    'csvs/dataset_organism_resistance_manualRemove_IG.csv',
    'csvs/dataset_organism_resistance_manualRemove_noUseless.csv',
    'csvs/dataset_organism_resistance_manualRemove_noUseless_wrapper.csv',
    'csvs/dataset_organism_resistance_noUseless.csv',
    'csvs/dataset_organism_resistance_noUseless_wrapper.csv',
    'csvs/dataset_organism_resistance.csv',
    'csvs/dataset_organism_resistance_IG.csv',
    'csvs/dataset_organism_resistance_manual.csv'
     ]
class_label = "organism_resistence"
classifiers = [MLPClassifier]
start = time.time()
for csv_file_path in csv_file_paths:
    with open(csv_file_path.split('/')[0]+'/result_{}.csv'.format(csv_file_path.split('/')[-1].split('.')[0]), 'a+') \
            as result_file_handler:
        fieldnames = ['classifier', 'fold', 'fname', 'precision', 'fscore', 'recall', 'auc', 'precision_b', 'fscore_b',
                      'recall_b', 'kappa', 'accuracy', 'tp_rate', 'micro_f', 'micro_p', 'micro_r']
        writer = csv.DictWriter(result_file_handler, fieldnames=fieldnames)
        writer.writeheader()
        print("============================= {} ============================".format(csv_file_path))
        data = pandas.read_csv(csv_file_path)
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns=['Unnamed: 0'])
        classes = data[class_label]
        data = data.drop(columns=[class_label])
        data = preprocess(data)
        classes = preprocess_classes(classes)

        # data, search_data, classes, search_classes = train_test_split(data, classes, test_size=.20, stratify=classes)
        #
        # search_iterations = 140
        # i = 0
        #
        # mean_std_pair = None
        # while i < len(classifiers):
        #     print("======= Param search {} ======".format(type(classifiers[i])))
        #     random_search = RandomizedSearchCV(classifiers[i], param_distributions=helper.PARAM_DISTS[type(classifiers[i])],
        #                                        n_iter=search_iterations, cv=5)
        #     mean = search_data.mean()
        #     std = search_data.std()
        #     search_data = normalize(search_data, mean, std)
        #     random_search.fit(search_data, search_classes)
        #     classifiers[i].set_params(**random_search.best_params_)
        #     i += 1
        # write_on_log(csv_file_path.replace('/', '_').split('.')[0], "========== Begin algorithm params {} =========".format(csv_file_path))
        # for classifier in classifiers:
        #     write_on_log(csv_file_path.replace('/', '_').split('.')[0], str(type(classifier)))
        #     write_on_log(csv_file_path.replace('/', '_').split('.')[0], str(classifier.get_params()))
        # write_on_log(csv_file_path.replace('/', '_').split('.')[0], "========== End algorithm params =========")

        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=15)
        results = []
        i = 0
        while i < len(classifiers):
            print("======= Training {} ======".format(type(classifiers[i]()).__name__))
            folds = 0
            for train_index, test_index in kf.split(data, classes):
                print("Fold {}".format(folds))
                classifier = classifiers[i]()
                params = helper.PARAM_DISTS[type(classifier).__name__]
                classifier.set_params(**params)
                data_train, data_test = data.iloc[train_index], data.iloc[test_index]
                mean = data_train.mean()
                std = data_train.std()
                data_train = normalize(data_train, mean, std)
                data_test = normalize(data_test, mean, std)
                classes_train, classes_test = classes[train_index], classes[test_index]
                kfold_result = dict()
                classifier.fit(data_train, classes_train)
                try:
                    predicted = classifier.predict(data_test)
                    metrics = dict()
                    metrics['fscore'] = f1_score(classes_test, predicted, average='weighted')
                    metrics['precision'] = precision_score(classes_test, predicted, average='weighted')
                    metrics['recall'] = recall_score(classes_test, predicted, average='weighted')
                    metrics['auc'] = roc_auc_score(classes_test, predicted, average='weighted')

                    metrics['micro_f'] = f1_score(classes_test, predicted, average='micro')
                    metrics['micro_p'] = precision_score(classes_test, predicted, average='micro')
                    metrics['micro_r'] = recall_score(classes_test, predicted, average='micro')

                    metrics['fscore_b'] = f1_score(classes_test, predicted)
                    metrics['precision_b'] = precision_score(classes_test, predicted)
                    metrics['recall_b'] = recall_score(classes_test, predicted)

                    metrics['kappa'] = cohen_kappa_score(classes_test, predicted)
                    metrics['accuracy'] = accuracy_score(classes_test, predicted)
                    tn, fp, fn, metrics['tp_rate'] = confusion_matrix(classes_test, predicted).ravel()
                    metrics['classifier'] = type(classifier).__name__
                    metrics['fold'] = folds
                    metrics['fname'] = csv_file_path.split('/')[-1]
                    results.append(metrics)
                    writer.writerow(metrics)
                    classifier_fname = './classifiers/{}_{}_fold{}.pkl'.format(csv_file_path.split('/')[-1].split('.')[0],
                                                                               type(classifiers[i]()).__name__, folds)
                    joblib.dump(classifier, classifier_fname)
                except Exception as e:
                    print(e)
                    kfold_result[type(classifier)] = 0
                folds += 1
            i+=1

elapsed = time.time() - start
time.strftime("%H:%M:%S", time.gmtime(elapsed))