import pkg_resources
pkg_resources.require("scikit-learn==0.19.1")
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import joblib
import shap
import os
from sklearn.model_selection import train_test_split, StratifiedKFold

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


explainer_fname = "explainer_best_proba2.pkl"
shap_values_fname = "shap_values_best_proba2.pkl"

print("Get classifiers csv")
classifiers_df = pd.read_csv('results/classifiers.csv')

print("Find best classifier")
best_classifier_row = classifiers_df[classifiers_df['kappa'] == max(classifiers_df['kappa'])].iloc[0]
# best_classifier_row = classifiers_df[ (classifiers_df['fname'] == 'dataset_organism_resistance_manual.csv')
#                                       & (classifiers_df['classifier'] == 'MLPClassifier')].iloc[0]
dataset = best_classifier_row['fname']
file = open('classifiers/{}'.format(best_classifier_row['classifier_fname']), 'rb')
best_classifier = joblib.load(file)

print("Reading dataset")
data = pd.read_csv('csvs/'+dataset)
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])
classes = data['organism_resistence']
print("Preprocessing dataset")
data = data.drop(columns=['organism_resistence'])
data = preprocess(data)
classes = preprocess_classes(classes)
columns = list(data.columns)

# data, search_data, classes, search_classes = train_test_split(data, classes, test_size=.90, stratify=classes, random_state=15)

print("Create kfold and loop through folds")
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=15)
kf = kf.split(data, classes)
fold = 0
while fold < best_classifier_row['fold']:
    t = next(kf)
    fold += 1
train_index, test_index = next(kf)
data_train, data_test = data.iloc[train_index], data.iloc[test_index]
print("Get mean and std")
mean = data_train.mean()
std = data_train.std()
print("Normalizing data")
data_train = normalize(data_train, mean, std)
data_test = normalize(data_test, mean, std)
classes_train, classes_test = classes[train_index], classes[test_index]

if os.path.exists(explainer_fname):
    explainer = joblib.load(open(explainer_fname, "rb"))
    shap_values = joblib.load(open(shap_values_fname, "rb"))
    print("Ploting")
    # shap.dependence_plot( "vasopressor_True",shap_values, data_test, columns)
    shap.summary_plot(shap_values, columns, plot_type="bar")
    # shap.force_plot(explainer.expected_value, shap_values[0], data_test[0], feature_names=columns, matplotlib=True)
    # plt.savefig('test.png')
    # print(shap_values)
else:
    print("Kmeans on data")
    data_train = shap.kmeans(data_train, 10)

    print("Creating exapliner")
    explainer = shap.KernelExplainer(best_classifier.predict_proba, data_train)
    print("Get shap values")
    shap_values = explainer.shap_values(data_test, l1_reg="num_features(10)")
    pickle.dump(explainer, open(explainer_fname, "wb"))
    pickle.dump(shap_values, open(shap_values_fname, "wb"))

    print("Ploting")
    shap.summary_plot(shap_values, columns)