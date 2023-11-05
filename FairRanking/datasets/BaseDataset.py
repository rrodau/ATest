import numpy as np
from sklearn.model_selection import train_test_split


def get_rand_chance(label_arr):
    if label_arr.ndim > 1:
        label_arr = label_arr[:, 0]
    _, counts = np.unique(label_arr, return_counts=True)
    return max(counts) / sum(counts)


class BaseDataset:

    def __init__(self, path):
        self.path = path
        self.got_data = False

    def path_to_file(self):
        raise NotImplementedError

    def load_data(self):
        raise NotImplementedError

    def get_data(self, random_state=42, test_size=0.2, val_size=0.2, val_split=True, preprocess_fn=None,
                 y_onehot=False):
        if not self.got_data:
            self.x_train, self.x_test, self.s_train, self.s_test, self.y_train, self.y_test = \
                train_test_split(self.x_col.values, self.s_col.values, self.y_col.values, test_size=test_size,
                                 random_state=random_state)
            if preprocess_fn is not None:
                self.x_train[:, self.numeric_cols] = preprocess_fn.fit_transform(self.x_train[:, self.numeric_cols])
                self.x_test[:, self.numeric_cols] = preprocess_fn.transform(self.x_test[:, self.numeric_cols])
            if val_split:
                self.x_train, self.x_val, self.s_train, self.s_val, self.y_train, self.y_val = \
                    train_test_split(self.x_train, self.s_train, self.y_train, test_size=val_size,
                                     random_state=random_state)
                if preprocess_fn is not None:
                    self.x_val[:, self.numeric_cols] = preprocess_fn.transform(self.x_val[:, self.numeric_cols])
            self.split_size = test_size
            self.y_train_rand_chance = get_rand_chance(self.y_train)
            self.s_train_rand_chance = get_rand_chance(self.s_train)
            self.y_test_rand_chance = get_rand_chance(self.y_test)
            self.s_test_rand_chance = get_rand_chance(self.s_test)
            if val_split:
                self.y_val_rand_chance = get_rand_chance(self.y_val)
                self.s_val_rand_chance = get_rand_chance(self.s_val)
            self.got_data = True
        if val_split:
            return (self.x_train, self.s_train, self.y_train), (self.x_val, self.s_val, self.y_val), \
                   (self.x_test, self.s_test, self.y_test)
        else:
            return (self.x_train, self.s_train, self.y_train), (self.x_test, self.s_test, self.y_test)

    def sensitive_attribute(self):
        raise NotImplementedError

    def get_num_fair_classes(self):
        return 2

    def get_num_features(self):
        return len(self.x_col.columns)

    def s_to_categorical(self, arr):
        return np.array(map(self.sensitive_attribute_map(arr)))

    def get_dataset_info(self):
        d = dict(vars(self))
        for key in ['df',
                    'x_col', 'y_col', 's_col',
                    'x_train', 'y_train', 's_train',
                    'x_test', 'y_test', 's_test',
                    'x_val', 'y_val', 's_val',
                    'toxicity_annotated_comments', 'toxicity_annotations']:
            try:
                d.pop(key)
            except KeyError:
                pass
        d["get_num_fair_classes"] = self.get_num_fair_classes()
        d["get_num_features"] = self.get_num_features()
        return d
