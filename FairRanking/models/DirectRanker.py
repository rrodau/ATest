import numpy as np
import tensorflow as tf

if tf.__version__.split('.')[0] == '2':
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()

from sklearn.base import BaseEstimator
from FairRanking.helpers import transform_pairwise
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import json
import pickle


class directRanker(BaseEstimator):
    """
    Constructor
    :param hidden_layers: List containing the numbers of neurons in the layers for feature
    :param feature_activation: tf function for the feature part of the net
    :param ranking_activation: tf function for the ranking part of the net
    :param feature_bias: boolean value if the feature part should contain a bias
    :param kernel_initializer: tf kernel_initializer
    :param start_batch_size: start value for increasing the sample size
    :param end_batch_size: end value for increasing the sample size
    :param learning_rate: learning rate for the optimizer
    :param max_steps: total training steps
    :param learning_rate_step_size: factor for increasing the learning rate
    :param learning_rate_decay_factor: factor for increasing the learning rate
    :param optimizer: tf optimizer object
    :param print_step: for which step the script should print out the cost for the current batch
    :param end_qids: end value for increasing the query size
    :param start_qids: start value for increasing the query size
    :param gamma: value how important the fair loss is
    """

    def __init__(self,
                 hidden_layers=[10, 5],
                 feature_activation=tf.nn.tanh,
                 ranking_activation=tf.nn.tanh,
                 feature_bias=True,
                 kernel_initializer=tf.random_normal_initializer(),
                 start_batch_size=100,
                 end_batch_size=500,
                 learning_rate=0.01,
                 max_steps=3000,
                 learning_rate_step_size=500,
                 learning_rate_decay_factor=0.944,
                 optimizer=tf.train.AdamOptimizer,
                 print_step=0,
                 end_qids=20,
                 start_qids=10,
                 random_seed=None,
                 name="DirectRanker",
                 tensor_name_prefix="",
                 sess=None
                 ):

        self.hidden_layers = hidden_layers
        self.feature_activation = feature_activation
        self.ranking_activation = ranking_activation
        self.feature_bias = feature_bias
        self.kernel_initializer = kernel_initializer
        self.start_batch_size = start_batch_size
        self.end_batch_size = end_batch_size
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.learning_rate_step_size = learning_rate_step_size
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.optimizer = optimizer
        self.print_step = print_step
        self.end_qids = end_qids
        self.start_qids = start_qids
        self.random_seed = random_seed
        self.name = name
        self.tensor_name_prefix = tensor_name_prefix
        self.sess = sess

    def _build_model(self, num_features, num_fair_features):
        """
        This function builds the directRanker with the values specified in the constructor
        :return:
        """
        assert self.tensor_name_prefix != ""

        # Placeholders for the inputs
        self.x0 = tf.placeholder(
            shape=[None, num_features],
            dtype=tf.float32,
            name=self.tensor_name_prefix + "x0"
        )
        self.x1 = tf.placeholder(
            shape=[None, num_features],
            dtype=tf.float32,
            name=self.tensor_name_prefix + "x1"
        )
        # Placeholder for the real classes
        self.y0 = tf.placeholder(
            shape=[None, 1],
            dtype=tf.float32,
            name=self.tensor_name_prefix + "y0"
        )

        # Constructing the feature creation part of the net
        nn0 = tf.layers.dense(
            inputs=self.x0,
            units=self.hidden_layers[0],
            activation=self.feature_activation,
            use_bias=self.feature_bias,
            kernel_initializer=self.kernel_initializer,
            name=self.tensor_name_prefix + "nn_hidden_0"
        )

        # By giving nn1 the same name as nn0 and using the flag reuse=True,
        # the weights and biases of all neurons in each branch are identical
        nn1 = tf.layers.dense(
            inputs=self.x1,
            units=self.hidden_layers[0],
            activation=self.feature_activation,
            use_bias=self.feature_bias,
            kernel_initializer=self.kernel_initializer,
            name=self.tensor_name_prefix + "nn_hidden_0",
            reuse=True
        )

        for i in range(1, len(self.hidden_layers)):
            nn0 = tf.layers.dense(
                inputs=nn0,
                units=self.hidden_layers[i],
                activation=self.feature_activation,
                use_bias=self.feature_bias,
                kernel_initializer=self.kernel_initializer,
                name=self.tensor_name_prefix + "nn_hidden_" + str(i)
            )
            nn1 = tf.layers.dense(
                inputs=nn1,
                units=self.hidden_layers[i],
                activation=self.feature_activation,
                use_bias=self.feature_bias,
                kernel_initializer=self.kernel_initializer,
                name=self.tensor_name_prefix + "nn_hidden_" + str(i),
                reuse=True
            )

        # Save last tensor of shared part to extract features later
        self.extracted_features = nn0

        # Creating antisymmetric features for the ranking
        self.nn = (nn0 - nn1) / 2.
        self.nn0 = nn0

        self.nn = tf.layers.dense(
            inputs=self.nn,
            units=1,
            activation=self.ranking_activation,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name=self.tensor_name_prefix + "nn_rank"
        )

        self.nn_cls = tf.layers.dense(
            inputs=nn0 / 2.,
            units=1,
            activation=self.ranking_activation,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name=self.tensor_name_prefix + "nn_rank",
            reuse=True
        )

        nn_out = tf.identity(
            input=self.nn,
            name=self.tensor_name_prefix + "nn"
        )

    def _cost(self, nn, y0):
        ranking_loss = (y0 - nn) ** 2
        return tf.reduce_mean(ranking_loss)

    def _build_pairs_compas(self, x, y, samples):
        """
        :param x:
        :param y:
        :param y_bias:
        :param samples:
        """
        x0 = []
        x1 = []
        y_train = []

        keys, counts = np.unique(y, return_counts=True)
        sort_ids = np.argsort(keys)
        keys = keys[sort_ids]
        counts = counts[sort_ids]
        for i in range(len(keys) - 1):
            indices0 = np.random.randint(0, counts[i + 1], samples)
            indices1 = np.random.randint(0, counts[i], samples)
            querys0 = np.where(y == keys[i + 1])[0]
            querys1 = np.where(y == keys[i])[0]
            x0.extend(x[querys0][indices0][:, :len(x)])
            x1.extend(x[querys1][indices1][:, :len(x)])
            y_train.extend((keys[i + 1] - keys[i]) * np.ones(samples))

        x0 = np.array(x0)
        x1 = np.array(x1)
        y_train = np.array([y_train]).transpose()

        return x0, x1, y_train

    def _build_pairs_trec(self, x, y, samples, around=30):
        """
        :param current_batch:
        :param around:
        """
        x0 = []
        x1 = []
        y_train = []

        keys, counts = np.unique(y, return_counts=True)
        sort_ids = np.argsort(keys)
        keys = keys[sort_ids]
        indices0 = np.random.randint(0, len(keys), samples)
        diff_indices1 = np.random.randint(-around, around, samples)
        indices1 = []
        for j in range(len(indices0)):
            if diff_indices1[j] == 0:
                diff_indices1[j] = 1
            tmp_idx = (indices0[j] + diff_indices1[j]) % len(keys)
            if tmp_idx > indices0[j]:
                indices1.append(indices0[j])
                indices0[j] = tmp_idx
            else:
                indices1.append(tmp_idx)
            assert indices0[j] > indices1[j]
        x0.extend([i for i in x[indices0]])
        x1.extend([i for i in x[indices1]])
        x0 = np.array(x0)
        x1 = np.array(x1)
        y_train.extend(1 * np.ones(samples))
        y_train = np.array([y_train]).transpose()

        return x0, x1, y_train

    def fit(self, x, y, y_bias, train_external=True, train_on_s=False, **fit_params):
        """
        :param features: list of queries for training the net
        :param real_classes: list of labels inside a query
        :param weights: list of weights per document inside a query
        :return:
        """

        if "num_features" not in fit_params.keys():
            raise AssertionError(
                '<num_features> Number of features are needed!'
            )

        if "num_fair_features" not in fit_params.keys():
            raise AssertionError(
                '<num_fair_features> Number of fair features are needed!'
            )

        if "dataset" not in fit_params.keys():
            raise AssertionError(
                '<dataset> Dataset ist needed!'
            )

        is_query_dataset = fit_params['dataset'] == 'trec'

        if train_on_s and is_query_dataset:
            y_new = []
            for yi in y:
                tmp = [yj[0] for yj in yi]
                y_new.append(tmp)
            y = np.array(y_new)

        self._build_model(fit_params["num_features"], fit_params["num_fair_features"])

        cost = self._cost(
            self.nn,
            self.y0
        )

        global_step = tf.Variable(0, trainable=False)
        increment_global_step = tf.assign(global_step, global_step + 1)
        learning_rate = tf.train.exponential_decay(
            self.learning_rate,
            global_step,
            self.learning_rate_step_size,
            self.learning_rate_decay_factor,
            staircase=True
        )

        optimizer = self.optimizer(learning_rate).minimize(
            cost,
            global_step=global_step
        )

        init = tf.global_variables_initializer()
        sample_factor = np.log(
            1.0 * self.end_batch_size / self.start_batch_size)
        q_factor = np.log(1.0 * self.end_qids / self.start_qids)

        if self.sess is None:
            self.sess = tf.Session()
        self.sess.run(init)

        if not is_query_dataset:

            for step in range(self.max_steps):
                samples = int(self.start_batch_size * np.exp(
                    1.0 * sample_factor * step / self.max_steps))

                x0, x1, y_train = self._build_pairs_compas(
                    x,
                    y,
                    samples
                )

                val, _, _ = self.sess.run(
                    [cost, optimizer, increment_global_step],
                    feed_dict={self.x0: x0,
                               self.x1: x1,
                               self.y0: y_train
                               })

                if self.print_step != 0 and step % self.print_step == 0:
                    print("step: {}, samples: {}".format(step, samples))

                    # print sub-losses for fair ranking
                    rank_loss_val = self.sess.run(
                        [cost],
                        feed_dict={self.x0: x0,
                                   self.x1: x1,
                                   self.y0: y_train
                                   })
                    print(
                        "ranking loss: {:7.5}".format(
                            rank_loss_val)
                    )

            if train_external:
                lr = LogisticRegression
                rf = RandomForestClassifier

                h_train = self.get_representations(x)
                # h_tmp, y_tmp, _ = transform_pairwise(h_train, y, subsample=0.5)
                s_train = np.array(y_bias).reshape(-1, 2)

                # self.lr_y = lr(solver='liblinear', multi_class='ovr').fit(h_tmp, y_tmp)
                # self.rf_y = rf(n_estimators=10).fit(h_tmp, y_tmp)

                self.lr_y = lr(solver='liblinear', multi_class='ovr').fit(h_train, y)
                self.rf_y = rf(n_estimators=10).fit(h_train, y)

                self.lr_s = lr(solver='liblinear', multi_class='ovr').fit(h_train, np.argmax(s_train, axis=1))
                self.rf_s = rf(n_estimators=10).fit(h_train, np.argmax(s_train, axis=1))

        else:

            for step in range(self.max_steps):
                samples = int(self.start_batch_size * np.exp(
                    1.0 * sample_factor * step / self.max_steps))

                q_samples = int(self.start_qids * np.exp(
                    1.0 * q_factor * step / self.max_steps))

                query_idx = np.random.choice(len(x), q_samples)

                for xis, yis, y_biasis in zip(x[query_idx], y[query_idx],
                                              y_bias[query_idx]):
                    if train_on_s:
                        x0, x1, y_train = self._build_pairs_compas(
                            xis,
                            yis,
                            samples
                        )
                    else:
                        x0, x1, y_train = self._build_pairs_trec(
                            xis,
                            yis,
                            samples
                        )

                    val, _, _ = self.sess.run(
                        [cost, optimizer, increment_global_step],
                        feed_dict={self.x0: x0,
                                   self.x1: x1,
                                   self.y0: y_train
                                   })

                if self.print_step != 0 and step % self.print_step == 0:
                    print(
                        "step: {}, samples: {}, queries: {}".format(
                            step, samples, q_samples))

                    # print sub-losses for fair ranking
                    rank_loss_val = self.sess.run(
                        [cost],
                        feed_dict={self.x0: x0,
                                   self.x1: x1,
                                   self.y0: y_train
                                   })
                    print(
                        "ranking loss: {}".format(
                            rank_loss_val)
                    )

            if train_external:
                lr = LogisticRegression
                rf = RandomForestClassifier

                h_all = []
                h_pair = []
                y_pair = []

                for i, x_query in enumerate(x):
                    h_tmp = self.get_representations(x_query)
                    h_all.append(h_tmp)
                    h_tmp, y_tmp, _ = transform_pairwise(h_tmp,
                                                         y[i],
                                                         subsample=0.1)
                    h_pair.append(h_tmp)
                    y_pair.append(y_tmp)

                h_train = np.array(h_pair).reshape(-1, h_tmp.shape[1])
                s_train = np.array(y_bias).reshape(-1, 2)
                y_train = np.array(y_pair).reshape(-1, 1)
                h_all = np.array(h_all).reshape(-1, h_tmp.shape[1])

                self.lr_y = lr(solver='liblinear', multi_class='ovr').fit(
                    h_train, y_train)
                self.rf_y = rf(n_estimators=10).fit(h_train, y_train)
                self.lr_s = lr(solver='liblinear', multi_class='ovr').fit(
                    h_all, np.argmax(s_train, axis=1))
                self.rf_s = rf(n_estimators=10).fit(h_all,
                                                    np.argmax(s_train,
                                                              axis=1))

        if fit_params['dataset'] == "german":
            raise AssertionError(
                'garman fit needs to be implemented'
            )

        if "save" in fit_params.keys():
            lr_y_model = 'lr_y_model.sav'
            rf_y_model = 'rf_y_model.sav'
            lr_s_model = 'lr_s_model.sav'
            rf_s_model = 'rf_s_model.sav'
            pickle.dump(self.lr_y, open(lr_y_model, 'wb'))
            pickle.dump(self.rf_y, open(rf_y_model, 'wb'))
            pickle.dump(self.lr_s, open(lr_s_model, 'wb'))
            pickle.dump(self.rf_s, open(rf_s_model, 'wb'))

    def predict_proba(self, features):
        """
        :param features: list of features of the instance feed to the net
        :return: predicted class
        """
        if len(features.shape) == 1:
            features = [features]

        res = self.sess.run(self.nn_cls, feed_dict={self.x0: features})

        return [0.5 * (value + 1) for value in res]

    def predict(self, features):
        """
        :param features: list of features of the instance feed to the net
        :return: predicted class
        """
        if len(features.shape) == 1:
            features = [features]

        res = self.sess.run(self.nn_cls, feed_dict={self.x0: features})

        prediction = []
        for value in res:
            if value > 0:
                prediction.append(0)
            else:
                prediction.append(1)

        return prediction

    def evaluate(self, x0, x1):
        """
        :param x0: pair of features
        :param x1: pair of features
        :return: predicted class
        """
        if len(x0.shape) == 1:
            x0 = [x0]
            x1 = [x1]
        return self.sess.run(self.nn, feed_dict={self.x0: x0, self.x1: x1})

    def get_representations(self, X):
        """
        Get the fair representations for X data.
        :param X: 2D numpy array containing original representations.
        :return: a 2D numpy array containing the fair representations.
        """
        features = self.sess.run(self.extracted_features, feed_dict={self.x0: X})
        return features

    def to_json(self, path, result_dict=None, fairness='unknown'):
        d = {'f': fairness}
        if result_dict is not None:
            for key in result_dict:
                d[key] = result_dict[key]
            for key in self.__dict__:
                d[key] = str(self.__dict__[key])
        else:
            d = {k: str(v) for k, v in self.__dict__.items()}
        with open(path, 'w') as f:
            json.dump(d, f, indent=4)

    @staticmethod
    def save(estimator, path):
        """

        :param path:
        :return:
        """
        saver = tf.train.Saver()
        if "/" not in path:
            path = "./" + path
        saver.save(estimator.sess, path + ".ckpt")

        save_dr = directRanker()
        for key in estimator.get_params():
            # ToDo: Need to be fixed to also restore the cost function
            if key == "cost":
                save_dr.__setattr__(key, None)
            else:
                save_dr.__setattr__(key, estimator.get_params()[key])

        with open(path + ".pkl", 'wb') as output:
            pickle.dump(save_dr, output, 0)

    @staticmethod
    def load_ranker(path):
        """

        :param path:
        :return:
        """
        # Some hack to solve the ValueError:
        # Variable nn_0/kernel/Adam/ already exists, disallowed. Did
        # you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?
        # https://github.com/kratzert/finetune_alexnet_with_tensorflow/issues/8
        tf.reset_default_graph()
        graph = tf.Graph()
        sess = tf.Session(graph=graph)
        with open(path + ".pkl", 'rb') as input:
            dr = pickle.load(input)

        with graph.as_default():
            saver = tf.train.import_meta_graph(path + ".ckpt.meta")
            saver.restore(sess, path + ".ckpt")
            dr.x0 = graph.get_tensor_by_name("x0:0")
            dr.x1 = graph.get_tensor_by_name("x1:0")
            dr.y0 = graph.get_tensor_by_name("y0:0")
            dr.nn = graph.get_tensor_by_name("nn:0")
        dr.sess = sess
        dr.num_features = dr.x0.shape[1].value

        return dr
