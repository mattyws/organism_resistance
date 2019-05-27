"""
Test classifiers generated by classification.py on respective data, to get the classification for each.
The classifier is selected by whom got its results near to the mean between folds
"""
import pkg_resources
from sklearn.model_selection import StratifiedKFold

pkg_resources.require("scikit-learn==0.19.1")
import pandas as pd
import joblib
import numpy as np

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
    return normalized_df

def preprocess_classes(classes):
    return np.array([0 if c == 'S' else 1 for c in classes])

print("Get classifiers csv")
class_label = "organism_resistence"
classifiers_df = pd.read_csv('results/classifiers.csv')
classifiers_df = classifiers_df[classifiers_df['kappa'] == max(classifiers_df['kappa'])]

datasets = classifiers_df['fname'].unique()

for dataset in datasets:
    print(" ==== {} ====".format(dataset))
    dataset_classifiers = classifiers_df[classifiers_df['fname'] == dataset]
    data = pd.read_csv('csvs/'+dataset)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    classes = data[class_label]
    data = data.drop(columns=[class_label])
    data = preprocess(data)
    classes = preprocess_classes(classes)
    # TODO : time is short so the k-fold mean and std is not generalized, do it after
    print("Looping k-fold")
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=15)
    kf = kf.split(data, classes)
    fold = 0
    while fold < 8:
        t = next(kf)
        fold += 1
    train_index, test_index = next(kf)
    data_train, data_test = data.iloc[train_index], data.iloc[test_index]
    print("Get mean and std")
    mean = data_train.mean()
    std = data_train.std()
    print("Normalizing data")
    data = normalize(data, mean, std)
    values = data.values
    for index, row in dataset_classifiers.iterrows():
        print(row['classifier'])
        result_fname = "classified_csv/{}_{}.csv".format(row['fname'], row['classifier'])
        classifier = joblib.load('classifiers/'+row['classifier_fname'])
        predictions = classifier.predict(values)
        new_dataset = data.copy(deep=True)
        new_dataset.loc[:, 'class'] = classes
        new_dataset.loc[:, 'prediction'] = predictions
        prediction_type = []
        for c, p in zip(classes, predictions):
            if c == 1 and c == p:
                prediction_type.append('TP')
            elif c == 1 and c != p:
                prediction_type.append('FN')
            elif c == 0 and c == p:
                prediction_type.append('TN')
            elif c == 0 and c != p:
                prediction_type.append('FP')
        new_dataset.loc[:, 'prediction_type'] = prediction_type
        new_dataset.to_csv(result_fname)

