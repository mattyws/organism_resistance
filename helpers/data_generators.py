import pickle
import uuid
import numpy as np
import os

import pandas
from keras.utils import Sequence

class LongitudinalDataGenerator(Sequence):

    def __init__(self, dataPaths, labels, batchSize, iterForever=False):
        self.batchSize = batchSize
        self.__labels = labels
        self.__filesList = dataPaths
        self.iterForever = iterForever
        self.__iterPos = 0

    def __load(self, filesNames):
        x = []
        y = []
        for fileName in filesNames:
            data = pandas.read_csv(fileName)
            if 'Unnamed: 0' in data.columns:
                data = data.drop(columns=['Unnamed: 0'])
            x.append(np.array(data.values))
        x = np.array(x)
        return x

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        batch_x = self.__filesList[idx * self.batchSize:(idx + 1) * self.batchSize]
        batch_x = self.__load(batch_x)
        batch_y = self.__labels[idx * self.batchSize:(idx + 1) * self.batchSize]
        return batch_x, batch_y

    def __len__(self):
        return np.int64(np.ceil(len(self.__filesList) / float(self.batchSize)))