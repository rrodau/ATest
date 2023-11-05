import numpy as np
import tensorflow as tf

if tf.__version__.split('.')[0] == '2':
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os
import pandas as pd


class BaseDirectRanker(BaseEstimator):
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
                 end_qids=300,
                 start_qids=10,
                 random_seed=42,
                 name="DirectRanker",
                 gamma=1.,
                 dataset=None,
                 noise_module=False,
                 noise_type='sigmoid_full',
                 whiteout=False,
                 uniform_noise=0,
                 whiteout_gamma=1.,
                 whiteout_lambda=1.,
                 num_features=0,
                 num_fair_classes=0,
                 save_dir=None):
        self.hidden_layers = hidden_layers
        self.feature_activation = feature_activation
        self.ranking_activation = ranking_activation
        self.feature_bias = feature_bias
        self.dataset = dataset
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
        self.gamma = gamma
        self.noise_module = noise_module
        self.noise_type = noise_type
        self.whiteout = whiteout
        assert not (self.whiteout and self.noise_module)
        self.whiteout_gamma = whiteout_gamma
        self.whiteout_lambda = whiteout_lambda
        self.uniform_noise = uniform_noise
        self.random_seed = random_seed
        self.name = name
        self.num_features = num_features
        self.num_fair_classes = num_fair_classes
        self.y_rankers = {'lr_y': {}, 'rf_y': {}}
        # 'lNet_y': {}, 'dr_y': {}}
        self.s_rankers = {'lr_s': {}, 'rf_s': {}}
        # 'dr_s': {}} #TODO: add back lNet_s
        self.checkpoint_steps = []
        if save_dir is None:
            self.save_dir = '/tmp/{}'.format(self.name)
        else:
            self.save_dir = save_dir
        self.load_path = self.save_dir + '/saved_model_step_{}'

    def _build_model(self):
        raise NotImplementedError

    def close_session(self):
        tf.reset_default_graph()
        self.sess.close()

    def create_feature_layers(self, input_0, input_1, hidden_size, name):
        nn0 = tf.layers.dense(
            inputs=input_0,
            units=hidden_size,
            activation=self.feature_activation,
            use_bias=self.feature_bias,
            kernel_initializer=self.kernel_initializer,
            name=name
        )

        nn1 = tf.layers.dense(
            inputs=input_1,
            units=hidden_size,
            activation=self.feature_activation,
            use_bias=self.feature_bias,
            kernel_initializer=self.kernel_initializer,
            name=name,
            reuse=True
        )

        return nn0, nn1

    def create_feature_layers_list_net(self, input, hidden_size, name):
        nn = tf.layers.dense(
            inputs=input,
            units=hidden_size,
            activation=self.feature_activation,
            use_bias=self.feature_bias,
            kernel_initializer=self.kernel_initializer,
            name=name
        )
        return nn

    def create_debias_layers(self, input_0, input_1, activation, name, num_units=20):
        nn_bias0 = tf.layers.dense(
            inputs=input_0,
            units=num_units,
            activation=activation,
            kernel_initializer=self.kernel_initializer,
            name=name
        )

        nn_bias1 = tf.layers.dense(
            inputs=input_1,
            units=num_units,
            activation=activation,
            kernel_initializer=self.kernel_initializer,
            name=name,
            reuse=True
        )

        return nn_bias0, nn_bias1

    def create_ranking_layers(self, input_0, input_1, return_aux_ranker=True):
        nn = (input_0 - input_1) / 2.

        nn = tf.layers.dense(
            inputs=nn,
            units=1,
            activation=self.ranking_activation,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name="nn_rank"
        )

        if return_aux_ranker:
            nn_cls = tf.layers.dense(
                inputs=input_0 / 2.,
                units=1,
                activation=self.ranking_activation,
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                name="nn_rank",
                reuse=True
            )
            return nn, nn_cls

        return nn

    def create_noise_module(self, input_0, input_1):
        with tf.variable_scope("%s-layer-%d" % ('noise', 0)):
            alpha = tf.get_variable("alpha", dtype=tf.float32,
                                    shape=[self.num_features],
                                    initializer=self.kernel_initializer)

            w_beta = tf.get_variable("w-beta", dtype=tf.float32,
                                     shape=[self.num_features],
                                     initializer=self.kernel_initializer)

            if self.uniform_noise == 1:
                noise = np.random.uniform(low=-1, high=1, size=self.num_features)
            else:
                noise = np.random.normal(size=self.num_features)
            noise = tf.constant(noise, dtype=np.float32)
            beta = tf.multiply(noise, w_beta)

            if self.noise_type == 'default':
                in_noise_0 = tf.multiply(input_0, alpha) + beta
                in_noise_1 = tf.multiply(input_1, alpha) + beta
            elif self.noise_type == 'sigmoid_full':
                in_noise_0 = tf.nn.sigmoid(tf.multiply(input_0, alpha) + beta)
                in_noise_1 = tf.nn.sigmoid(tf.multiply(input_1, alpha) + beta)
            elif self.noise_type == 'sigmoid_sep':
                in_noise_0 = tf.nn.sigmoid(tf.multiply(input_0, alpha)) + tf.nn.sigmoid(beta)
                in_noise_1 = tf.nn.sigmoid(tf.multiply(input_1, alpha)) + tf.nn.sigmoid(beta)
            elif self.noise_type == 'sigmoid_sep_2':
                in_noise_0 = (tf.nn.sigmoid(tf.multiply(input_0, alpha)) + tf.nn.sigmoid(beta)) / 2
                in_noise_1 = (tf.nn.sigmoid(tf.multiply(input_1, alpha)) + tf.nn.sigmoid(beta)) / 2

        return in_noise_0, in_noise_1

    def create_noise_module_list_net(self, input):
        with tf.variable_scope("%s-layer-%d" % ('noise', 0)):
            alpha = tf.get_variable("alpha", dtype=tf.float32,
                                    shape=[self.num_features],
                                    initializer=self.kernel_initializer)

            w_beta = tf.get_variable("w-beta", dtype=tf.float32,
                                     shape=[self.num_features],
                                     initializer=self.kernel_initializer)

            if self.uniform_noise == 1:
                noise = np.random.uniform(low=-1, high=1, size=self.num_features)
            else:
                noise = np.random.normal(size=self.num_features)
            noise = tf.constant(noise, dtype=np.float32)
            beta = tf.multiply(noise, w_beta)

            if self.noise_type == 'default':
                in_noise = tf.multiply(input, alpha) + beta
            elif self.noise_type == 'sigmoid_full':
                in_noise = tf.nn.sigmoid(tf.multiply(input, alpha) + beta)
            elif self.noise_type == 'sigmoid_sep':
                in_noise = tf.nn.sigmoid(tf.multiply(input, alpha)) + tf.nn.sigmoid(beta)
            elif self.noise_type == 'sigmoid_sep_2':
                in_noise = (tf.nn.sigmoid(tf.multiply(input, alpha)) + tf.nn.sigmoid(beta)) / 2

        return in_noise

    def fit(self, x, y, y_bias, **fit_params):
        is_query_dataset = fit_params['dataset'] == 'trec'
        ckpt_period = fit_params['ckpt_period']
        train_external = fit_params['train_external']

        self._build_model()

        ranking_cost = self.ranking_cost
        if self.gamma != 0:
            fair_cost = self.fair_cost
        else:
            fair_cost = tf.no_op()

        global_step = tf.Variable(0, trainable=False)
        increment_global_step = tf.assign(global_step, global_step + 1)
        learning_rate = tf.train.exponential_decay(
            self.learning_rate,
            global_step,
            self.learning_rate_step_size,
            self.learning_rate_decay_factor,
            staircase=True
        )

        ranking_optimizer = self.optimizer(learning_rate).minimize(
            ranking_cost,
            global_step=global_step
        )

        if self.gamma != 0:
            fair_optimizer = self.optimizer(learning_rate).minimize(
                fair_cost,
                global_step=global_step
            )
        else:
            fair_optimizer = tf.no_op()

        if 'adv' in self.name.lower() and 'flip' in self.name.lower():
            fair_cost_extra = self.fair_cost_extra
            fair_optimizer_extra = self.optimizer(learning_rate).minimize(
                fair_cost,
                global_step=global_step
            )

        init = tf.global_variables_initializer()
        sample_factor = np.log(
            1.0 * self.end_batch_size / self.start_batch_size)
        q_factor = np.log(1.0 * self.end_qids / self.start_qids)

        self.sess = tf.Session()
        tf.set_random_seed(self.random_seed)  # Set random seed for reproducibility.
        self.sess.run(init)
        self.saver = tf.train.Saver(max_to_keep=0)

        for step in range(self.max_steps):
            num_samples = int(self.start_batch_size * np.exp(
                1.0 * sample_factor * step / self.max_steps))
            """TODO: at the moment now query fitting"""
            if not is_query_dataset:
                feed_dict = self._get_feed_dict(x, y, y_bias, num_samples)
            else:
                feed_dict = self._get_feed_dict_queries(x, y, y_bias, num_samples)
            rank_loss_val, _, _ = self.sess.run(
                [ranking_cost, ranking_optimizer, increment_global_step],
                feed_dict=feed_dict)
            fair_loss_val, _, _ = self.sess.run(
                [fair_cost, fair_optimizer, increment_global_step],
                feed_dict=feed_dict)
            if fair_loss_val is None:
                fair_loss_val = 0.
            if 'adv' in self.name.lower() and 'flip' in self.name.lower():
                fair_extra_loss_val, _, _ = self.sess.run(
                    [fair_cost_extra, fair_optimizer_extra, increment_global_step],
                    feed_dict=feed_dict)
            if step % ckpt_period == 0:
                if train_external:
                    self.train_external_rankers(x, y, y_bias, step, is_query_dataset)
                self.save_model(step)
                self.checkpoint_steps.append(step)
                if self.print_step > 0:
                    print("step: {}, samples: {}".format(step, num_samples))
                    if 'adv' in self.name.lower() and 'flip' in self.name.lower():
                        print(
                            "loss: {:7.5} fair loss: {:7.5} fair loss 2: {:7.5} ranking loss: {:7.5}".format(
                                self.gamma * fair_loss_val + rank_loss_val + self.gamma * fair_extra_loss_val,
                                fair_loss_val,
                                rank_loss_val,
                                fair_extra_loss_val)
                        )
                    else:
                        print(
                            "loss: {:7.5} fair loss: {:7.5} ranking loss: {:7.5}".format(
                                self.gamma * fair_loss_val + rank_loss_val,
                                fair_loss_val,
                                rank_loss_val)
                        )
        if train_external:
            self.train_external_rankers(x, y, y_bias, 'final', is_query_dataset)

    def train_external_rankers(self, x, y, s_train, step, is_query_dataset):
        lr = LogisticRegression
        rf = RandomForestClassifier

        # the classifiers below require non-one-hot encoding. so we convert only if needed.
        if s_train[0].shape[0] > 1:
            # this conversion is a placeholder and ideally I would like to have the dataset object
            # involved somehow. however that requires a lot of information to be passed through here,
            # the dataset name at the very least. this does work, but it hides the assumption that
            # the data is [1, 0] -> protected; [0, 1] -> unprotected which is a little dirty.
            s_train = s_train[:, 0]

        dataset = 'trec' if is_query_dataset == True else 'foo'

        h_train = self.get_representations(x)

        lr_y = lr(solver='liblinear', multi_class='ovr').fit(h_train, y)
        rf_y = rf(n_estimators=10).fit(h_train, y)

        lr_s = lr(solver='liblinear', multi_class='ovr').fit(h_train, s_train)
        rf_s = rf(n_estimators=10).fit(h_train, s_train)

        self.y_rankers['lr_y'][step] = lr_y
        self.s_rankers['lr_s'][step] = lr_s
        self.y_rankers['rf_y'][step] = rf_y
        self.s_rankers['rf_s'][step] = rf_s

    def save_model(self, step):
        path = '{}/saved_model_step_{}'.format(self.save_dir, step)
        os.makedirs(path)
        self.saver.save(self.sess, path + '/model.ckpt')

    def load_model(self, step):
        """
        Note that because of the first line this function is not parallel-safe.
        """
        tf.reset_default_graph()
        self.sess = tf.Session()
        self._build_model()
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.load_path.format(step) + '/model.ckpt')

    def get_representations(self, x):
        return self.sess.run(self.extracted_representations, feed_dict={self.x0: x})

    def _get_feed_dict(self, x, y, y_bias, samples):
        """
        :param x:
        :param y:
        :param y_bias:
        :param samples:
        """
        x0 = []
        x1 = []
        y_train = []
        y0_bias = []
        y1_bias = []
        y_bias_train = []
        # y_bias = y_bias[:, 0]

        keys, counts = np.unique(y, return_counts=True)
        sort_ids = np.argsort(keys)
        keys = keys[sort_ids]
        counts = counts[sort_ids]
        for i in range(len(keys) - 1):
            indices0 = np.random.randint(0, counts[i + 1], samples)
            indices1 = np.random.randint(0, counts[i], samples)
            querys0 = np.where(y == keys[i + 1])[0]
            querys1 = np.where(y == keys[i])[0]
            x0.extend(x[querys0][indices0])
            x1.extend(x[querys1][indices1])
            y_train.extend((keys[i + 1] - keys[i]) * np.ones(samples))
            y0_bias.extend(y_bias[querys0][indices0])
            y1_bias.extend(y_bias[querys1][indices1])
            tmp_0 = np.array(y_bias[querys0][indices0])
            tmp_1 = np.array(y_bias[querys1][indices1])
            y_bias_train.extend(tmp_0 - tmp_1)

        x0 = np.array(x0)
        x1 = np.array(x1)
        y_bias_train = np.array(y_bias_train)[:, 0]
        y_train = np.array([y_train]).transpose()
        y0_bias = np.array(y0_bias)
        y1_bias = np.array(y1_bias)
        y_bias_train = np.expand_dims(y_bias_train, axis=1)
        _, y_bias_counts = np.unique(y_bias_train, return_counts=True)
        if np.shape(y_bias_counts)[0] == 1:
            num_nonzeros = 1
        else:
            num_nonzeros = y_bias_counts[1]
        adj_sym_loss_term = len(x0) / num_nonzeros

        return {'x0': x0, 'x1': x1, 'y_train': y_train,
                'y_bias_0': y0_bias, 'y_bias_1': y1_bias,
                'y_bias': y_bias_train, 'adj_loss_term': adj_sym_loss_term}

    def _get_feed_dict_google(self, pairs_dict, scores):

        y_bias_0 = pairs_dict['y_bias_0']
        y_bias_1 = pairs_dict['y_bias_1']

        y_train = []
        y_train.extend(np.ones(int(len(y_bias_0))))
        y_train.extend(0 * np.ones(int(len(y_bias_1))))
        y_train = np.array(y_train)

        y_bias = []
        y_bias.extend(y_bias_0)
        y_bias.extend(y_bias_1)
        y_bias = np.array(y_bias)

        df = pd.DataFrame()
        df = df.assign(scores=scores, labels=y_train, groups=y_bias[:, 0], merge_key=0)
        df = df.merge(df.copy(), on="merge_key", how="outer", suffixes=("_high", "_low"))
        df = df[df.labels_high > df.labels_low]

        paired_scores = np.stack([df.scores_high.values, df.scores_low.values], axis=1)
        paired_groups = np.stack([df.groups_high.values, df.groups_low.values], axis=1)

        sub0 = (paired_groups[:, 0] == 0) & (paired_groups[:, 1] == 1)
        sub1 = (paired_groups[:, 0] == 1) & (paired_groups[:, 1] == 0)

        return {self.scores_tensor: paired_scores.T.reshape(-1, ),
                self.subset0_predicate: sub0, self.subset1_predicate: sub1}

    def _get_feed_dict_queries(self, x, y, y_bias, samples, around=30):
        """
        :param current_batch:
        :param around:
        """
        x0 = []
        x1 = []
        y_train = []
        y0_bias = []
        y1_bias = []

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

        y0_bias.extend(y_bias[indices0])
        y1_bias.extend(y_bias[indices1])
        y0_bias = np.array(y0_bias)
        y1_bias = np.array(y1_bias)

        return {'x0': x0, 'x1': x1, 'y_train': y_train,
                'y0_bias': y0_bias, 'y1_bias': y1_bias}

    def get_lr_y_ranker(self):
        return self.y_rankers['lr_y']['final']

    def get_lr_s_ranker(self):
        return self.s_rankers['lr_s']['final']

    def get_rf_y_ranker(self):
        return self.y_rankers['rf_y']['final']

    def get_rf_s_ranker(self):
        return self.s_rankers['rf_s']['final']

    def to_dict(self):
        """
        Return a dictionary representation of the object while dropping the tensorflow stuff.
        Useful to keep track of hyperparameters at the experiment level.
        """
        d = dict(vars(self))
        for key in ['y_rankers', 's_rankers', 'sess', 'x0', 'x1', 'y', 'x', 'y_fair_0', 'y_fair_1', 'nn_group_0',
                    'nn_group_1', 'y_bias_0', 'y_bias_1', 'extracted_features', 'nn_bias0', 'nn_bias1', 'nn',
                    'nn_cls', 'fair_cost', 'ranking_cost', 'y_bias', 'nn_bias', 'fair_cost_extra', 'nn_bias_0',
                    'nn_bias_1', 'saver', 'feature_activation', 'ranking_activation', 'kernel_initializer', 'optimizer',
                    'load_path', 'adj_loss_term', 'nn_bias_cls'
                    ]:
            try:
                d.pop(key)
            except KeyError:
                pass
        return d

    def predict_proba(self, features):
        """
        :param features: list of features of the instance feed to the net
        :return: predicted class
        """
        if len(features.shape) == 1:
            features = [features]

        if self.name == "FairListNet":
            res = self.sess.run(self.nn_cls, feed_dict={self.x: features})
        else:
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
        if self.name == "FairListNet":
            features = self.sess.run(self.extracted_features, feed_dict={self.x: X})
        else:
            features = self.sess.run(self.extracted_features, feed_dict={self.x0: X})

        return features


def create_placeholder(num_features):
    placeholder = tf.placeholder(
        shape=[None, num_features],
        dtype=tf.float32,
        name="placeholder"
    )
    return placeholder


def create_scalar_placeholder():
    placeholder = tf.placeholder(
        shape=(),
        dtype=tf.float32,
        name="scalar_placeholder"
    )
    return placeholder


def build_pairs(x, y, y_bias, samples):
    """
    :param x:
    :param y:
    :param y_bias:
    :param samples:
    """
    x0 = []
    x1 = []
    y_train = []
    y0_bias = []
    y1_bias = []

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
        y0_bias.extend(y_bias[querys0][indices0])
        y1_bias.extend(y_bias[querys1][indices1])

    x0 = np.array(x0)
    x1 = np.array(x1)
    y_train = np.array([y_train]).transpose()
    y0_bias = np.array(y0_bias)
    y1_bias = np.array(y1_bias)

    return x0, x1, y_train, y0_bias, y1_bias


def build_pairs_query(self, x, y, y_bias, samples, around=30):
    """
    :param current_batch:
    :param around:
    """
    x0 = []
    x1 = []
    y_train = []
    y0_bias = []
    y1_bias = []

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

    y0_bias.extend(y_bias[indices0])
    y1_bias.extend(y_bias[indices1])
    y0_bias = np.array(y0_bias)
    y1_bias = np.array(y1_bias)

    return x0, x1, y_train, y0_bias, y1_bias
