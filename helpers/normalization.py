import csv

import pandas as pd
import numpy as np

class Normalization(object):

    def __init__(self, normalization_values, temporary_path='./data_tmp/'):
        self.normalization_values = normalization_values
        self.temporary_path = temporary_path

    def normalize_files(self, filesList):
        """
        Normalize all files in a list of paths
        :param filesList: the list of files path
        :return: a new list for the paths of the normalized data
        """
        newList = []
        for file in filesList:
            fileName = file.split('/')[-1]
            data = pd.read_csv(file)
            if 'Unnamed: 0' in data.columns:
                data = data.drop(columns=['Unnamed: 0'])
            data = self.__normalize(data)
            # Sort columns
            columns = list(data.columns)
            data.to_csv(self.temporary_path+fileName, index=False)
            newList.append(self.temporary_path+fileName)
        return newList


    def __normalize(self, data):
        """
        Normalize data using the normalization_values (max and min for the column)
        :param data: the data to be normalized
        :return: the data normalized
        """
        for column in data.columns:
            data.loc[:, column] = self.__z_score_normalization(column, data[column])
        return data

    def __min_max_normalization(self, column, series):
        max = self.normalization_values[column]['max']
        min = self.normalization_values[column]['min']
        return series.apply(lambda x: (x - min) / (max - min))

    def __z_score_normalization(self, column, series):
        # If std is equal to 0, all columns have the same value
        if self.normalization_values[column]['std'] != 0:
            mean = self.normalization_values[column]['mean']
            std = self.normalization_values[column]['std']
            return series.apply(lambda x: (x - mean) / std)
        return series

    @staticmethod
    def get_normalization_values(filesList):
        """
        Get the max and min value for each column from a set of csv files
        :param filesList: the list of files to get the value
        :return: a dict with the max and min value for each column
        """
        values = dict()
        # Loop each file in dataset
        for file in filesList:
            df = pd.read_csv(file)
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
            # Loop each column in file
            for column in df.columns:
                # Add if column don't exist at keys
                if column not in values.keys():
                    values[column] = dict()
                    values[column]['values'] = pd.Series([])
                    # values[column]['max'] = None
                    # values[column]['min'] = None
                values[column]['values'] = values[column]['values'].append(df[column])
                # Get max and min values for the column
                # max = df[column].max()
                # min = df[column].min()
                # Replace values for max and min for this column
                # if values[column]['max'] is None or values[column]['max'] < max:
                #     values[column]['max'] = max
                # if values[column]['min'] is None or values[column]['min'] > min:
                #     values[column]['min'] = min
        for key in values.keys():
            values[key]['max'] = values[key]['values'].max()
            values[key]['min'] = values[key]['values'].min()
            values[key]['mean'] = values[key]['values'].mean()
            values[key]['std'] = values[key]['values'].std()
            values[key]['values'] = None
        return values
