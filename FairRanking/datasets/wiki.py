from FairRanking.datasets.BaseDataset import BaseDataset, get_rand_chance
import pandas as pd
import numpy as np

pd.set_option('display.max_colwidth', 1000)
import tensorflow_hub as hub
import tensorflow as tf

if tf.__version__.split('.')[0] == '2':
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()


class Wiki(BaseDataset):

    def __init__(self, path="data/", embedding="https://tfhub.dev/google/Wiki-words-250/2"):
        super().__init__(path=path)
        self.embedding = embedding
        self.name = 'wiki'
        self.one_hot_cols = [3, 5, 6, 7, 8, 9]
        self.discard_cols = [1, 13]
        self.label_col = [14]
        self.numeric_cols = list(set(range(15)) - set(self.one_hot_cols) - set(self.label_col) - set(self.discard_cols))
        self.load_data()

    def path_to_file(self):
        return self.path + '/toxicity_annotated_comments.tsv', self.path + '/toxicity_annotated.tsv',

    def filter_frame(self, frame, keyword=None, length=None):
        """Filters DataFrame to comments that contain the keyword as a substring and fit within length."""
        if keyword:
            frame = frame[frame['comment'].str.contains(keyword, case=False)]
        if length:
            frame = frame[frame['length'] <= length]
        return frame

    def class_balance(self, frame, keyword, length=None):
        """Returns the fraction of the dataset labeled toxic."""
        frame = self.filter_frame(frame, keyword, length)
        return len(frame.query('is_toxic')) / len(frame)

    def load_data(self, num_classes=10, subsample=0.1, seed=42):
        self.toxicity_annotated_comments = pd.read_csv(self.path_to_file()[0], sep='\t')
        self.toxicity_annotations = pd.read_csv(self.path_to_file()[1], sep='\t')
        annotations_gped = self.toxicity_annotations.groupby('rev_id', as_index=False).agg({'toxicity': 'mean'})
        all_data = pd.merge(annotations_gped, self.toxicity_annotated_comments, on='rev_id')

        all_data['comment'] = all_data['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
        all_data['comment'] = all_data['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
        all_data['length'] = all_data['comment'].str.len()

        # TODO(nthain): Consider doing regression instead of classification
        all_data['is_toxic'] = all_data['toxicity'] > 0.5
        all_data['is_gay'] = all_data['comment'].str.contains('gay', case=False)

        for header in all_data.head():
            if header not in ["comment", "is_toxic", "logged_in", "length", "is_gay", "ns", "rev_id", "sample", "split",
                              "toxicity", "year"]:
                del all_data[header]

        embed = hub.load(self.embedding)
        embed = embed(all_data["comment"])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            embedding_x = sess.run(embed)
        tf.reset_default_graph()
        self.x_col = pd.DataFrame(data=np.array(embedding_x))
        self.y_col = all_data.loc[:, "is_toxic"].astype(int)
        self.s_col = pd.DataFrame(data=np.array([[v, abs(v - 1)] for v in all_data["is_gay"]]))
        self.num_relevance_classes = len(np.unique(self.y_col.values))

    def sensitive_attribute(self):
        return 'gay'

    @staticmethod
    def class_attribute():
        return 'ZFYA'
