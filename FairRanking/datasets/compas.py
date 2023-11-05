from FairRanking.datasets.BaseDataset import BaseDataset, get_rand_chance
import pandas as pd
import numpy as np


class Compas(BaseDataset):

    def __init__(self, path="data/"):
        super().__init__(path=path)
        self.load_data()
        self.name = 'compas'
        self.numeric_cols = [-1]
        self.one_hot_cols = list(range(7))

    def path_to_file(self):
        return self.path + '/compas_preprocessed_ranking_2races.csv'

    def load_data(self):
        self.df = pd.read_csv(self.path_to_file())
        self.y_col = self.df.loc[:, self.class_attribute()]
        s_name = [col_name for col_name in self.df.columns if col_name.startswith(self.sensitive_attribute())]
        self.s_col = self.df.loc[:, s_name]
        x_name = set(self.df.columns) - set([self.class_attribute()]) - set(s_name) - set([self.discard_columns()])
        self.x_col = self.df.loc[:, x_name]
        self.num_features = len(self.x_col.columns)
        self.num_relevance_classes = len(np.unique(self.y_col.values))

    def sensitive_attribute(self):
        return 'race'

    def class_attribute(self):
        return 'decile_score'

    def discard_columns(self):
        return 'score_text'

    def sensitive_attribute_map(self):
        return {[0, 1]: 0, [1, 0]: 1}

    def sensitive_attribute_map_str(self):
        return {[0, 1]: 'White', [1, 0]: 'Black'}

    def s_to_categorical(self, arr):
        return arr[:, 0].reshape(-1, 1)


if __name__ == '__main__':
    print('pippo')
