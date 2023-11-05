from FairRanking.datasets.BaseDataset import BaseDataset, get_rand_chance
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Law(BaseDataset):

    def __init__(self, sensitive_str, path="data/"):
        super().__init__(path=path)
        if sensitive_str == 'race':
            self.sensitive_attribute_str = 'race'
        elif sensitive_str == 'gender':
            self.sensitive_attribute_str = 'gender'
        else:
            raise ValueError('sensitive_str {} does not match "gender" or "race".'.format(sensitive_str))
        self.name = 'law-' + self.sensitive_attribute_str
        self.load_data()

    def path_to_file(self):
        return self.path + '/law_data.csv'

    def load_data(self, num_classes=10, subsample=0.1, seed=42):
        self.df = pd.read_csv(self.path_to_file())
        excluded_s = ['gender', 'race']
        self.df, _ = train_test_split(self.df, test_size=1 - subsample,
                                      stratify=self.df.loc[:, self.sensitive_attribute()],
                                      random_state=seed)
        self.df = self.df[(self.df.race == 'White') | (self.df.race == 'Black')]
        # from https://github.com/citlab/Meike-FairnessInL2R-Code/blob/master/src/data_preparation/lawStudentDatasetPreparation.py#L27
        self.df['gender'] = self.df['gender'].replace([2], 0)
        self.df = self.df.sort_values(by=self.class_attribute())
        self.df['position'] = range(len(self.df), 0, -1)
        self.df['position'] /= len(self.df)
        self.df['position'] *= num_classes
        self.df['position'] = np.floor(self.df['position'] - 1e-12)
        if self.sensitive_attribute() == 'gender':
            self.s_col = pd.DataFrame(
                np.array([self.replace_gender(val) for val in self.df[self.sensitive_attribute()]]))
        else:
            self.s_col = pd.DataFrame(np.array([self.replace_race(val) for val in self.df[self.sensitive_attribute()]]))
        self.y_col = self.df.loc[:, 'position']
        x_name = set(self.df.columns) - set(self.class_attribute()) - set(excluded_s) - set(self.discard_columns())
        self.x_col = self.df.loc[:, x_name]
        self.numeric_cols = list(range(len(x_name)))
        self.num_relevance_classes = len(np.unique(self.y_col.values))

    @staticmethod
    def discard_columns():
        return ['region_first', 'sander_index', 'first_pf', 'id', 'position']

    @staticmethod
    def replace_gender(value):
        if value == 0:
            return [0, 1]
        elif value == 1:
            return [1, 0]
        else:
            raise ValueError('Non-binary gender attribute.')

    @staticmethod
    def replace_race(value):
        if value == 'White':
            return [0, 1]
        elif value == 'Black':
            return [1, 0]
        else:
            raise ValueError('Non-binary race attribute')

    def sensitive_attribute(self):
        return self.sensitive_attribute_str

    @staticmethod
    def class_attribute():
        return 'ZFYA'
