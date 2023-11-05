from FairRanking.datasets.BaseDataset import BaseDataset, get_rand_chance
import pandas as pd
import numpy as np


class Adult(BaseDataset):

    def __init__(self, path="data/"):
        super().__init__(path=path)
        self.name = 'adult'
        self.one_hot_cols = [3, 5, 6, 7, 8, 9]
        self.discard_cols = [1, 13]
        self.label_col = [14]
        self.numeric_cols = list(set(range(15)) - set(self.one_hot_cols) - set(self.label_col) - set(self.discard_cols))
        self.load_data()

    def path_to_file(self):
        return self.path + '/adult.csv'

    def load_data(self, num_classes=10, subsample=0.1, seed=42):
        self.df = pd.read_csv(self.path_to_file(), header=None)
        # drop columns
        self.df = self.df.drop(self.discard_cols, axis=1)
        # get one hot encodings
        self.df = pd.get_dummies(self.df, columns=self.one_hot_cols)
        # map labels into 0, 1
        self.df[14].replace(['<=50K', '>50K'], [0., 1.], inplace=True, regex=True)
        # split into x, y, s
        sensible_cols = ['9_ Female', '9_ Male']
        self.x_col = self.df[self.df.columns.difference(self.label_col + sensible_cols)]
        self.y_col = self.df[self.label_col]
        self.s_col = self.df[sensible_cols].astype('int64')
        self.num_relevance_classes = len(np.unique(self.y_col.values))

    def sensitive_attribute(self):
        return 'race'

    @staticmethod
    def class_attribute():
        return 'ZFYA'
