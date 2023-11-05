import tensorflow as tf

if tf.__version__.split('.')[0] == '2':
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
import numpy as np
from FairRanking.models.BaseDirectRanker import BaseDirectRanker, create_scalar_placeholder, create_placeholder


class FairListNet(BaseDirectRanker):
    """
    Tensorflow implementation of https://arxiv.org/pdf/1805.08716.pdf
    Inspired by: https://github.com/MilkaLichtblau/DELTR-Experiments

    Constructor
    :param hidden_layers: List containing the numbers of neurons in the layers for feature
    :param activation: tf function for the feature part of the net
    :param kernel_initializer: tf kernel_initializer
    :param start_batch_size: cost function of FairListNet
    :param min_doc: min size of docs in query if a list is given
    :param end_batch_size: max size of docs in query if a list is given
    :param start_len_qid: start size of the queries/batch
    :param end_len_qid: end size of the queries/batch
    :param learning_rate: learning rate for the optimizer
    :param max_steps: total training steps
    :param learning_rate_step_size: factor for increasing the learning rate
    :param learning_rate_decay_factor: factor for increasing the learning rate
    :param optimizer: tf optimizer object
    :param print_step: for which step the script should print out the cost for the current batch
    :param weight_regularization: float for weight regularization
    :param dropout: float amount of dropout
    :param input_dropout: float amount of input dropout
    :param name: name of the object
    :param num_features: number of input features
    :param protected_feature_deltr: column name of the protected attribute (index after query and document id)
    :param gamma_deltr: value of the gamma parameter
    :param iterations_deltr: number of iterations the training should run
    :param standardize_deltr: let's apply standardization to the features
    :param random_seed: random seed
    """

    def __init__(self,
                 hidden_layers=[10],
                 feature_activation=tf.nn.tanh,
                 ranking_activation=tf.nn.tanh,
                 kernel_initializer=tf.random_normal_initializer(),
                 start_batch_size=100,
                 end_batch_size=3000,
                 start_qids=10,
                 end_qids=100,
                 learning_rate=0.01,
                 max_steps=3000,
                 dataset=None,
                 max_queries=50,
                 learning_rate_step_size=500,
                 learning_rate_decay_factor=0.944,
                 optimizer=tf.train.AdamOptimizer,
                 print_step=0,
                 name="FairListNet",
                 gamma=1,
                 feature_bias=True,
                 noise_module=False,
                 noise_type='sigmoid_full',
                 whiteout=False,
                 uniform_noise=0,
                 whiteout_gamma=1.,
                 whiteout_lambda=1.,
                 num_features=0,
                 num_fair_classes=0,
                 random_seed=42,
                 save_dir=None
                 ):
        super().__init__(hidden_layers=hidden_layers, dataset=dataset,
                         feature_activation=feature_activation, ranking_activation=ranking_activation,
                         feature_bias=feature_bias, kernel_initializer=kernel_initializer,
                         start_batch_size=start_batch_size, end_batch_size=end_batch_size,
                         learning_rate=learning_rate, max_steps=max_steps,
                         learning_rate_step_size=learning_rate_step_size,
                         learning_rate_decay_factor=learning_rate_decay_factor, optimizer=optimizer,
                         print_step=print_step,
                         end_qids=end_qids, start_qids=start_qids, random_seed=random_seed, name=name, gamma=gamma,
                         noise_module=noise_module, noise_type=noise_type, whiteout=whiteout,
                         uniform_noise=uniform_noise,
                         whiteout_gamma=whiteout_gamma, whiteout_lambda=whiteout_lambda, num_features=num_features,
                         num_fair_classes=num_fair_classes, save_dir=save_dir)

    def _build_model(self):
        """
        This function builds the ListNet with the values specified in the constructor
        :return:
        """

        self.x = create_placeholder(self.num_features)
        self.y = create_placeholder(1)
        self.y_fair_0 = tf.placeholder(shape=[None], dtype=tf.int32, name="y_fair_0")
        self.y_fair_1 = tf.placeholder(shape=[None], dtype=tf.int32, name="y_fair_1")

        if self.noise_module:
            in_x = self.create_noise_module_list_net(self.x)
        else:
            in_x = self.x

        for i, num_nodes in enumerate(self.hidden_layers):
            in_x = self.create_feature_layers_list_net(in_x, num_nodes, 'nn_hidden_{}'.format(i))

        self.extracted_features = in_x

        self.nn = tf.layers.dense(
            inputs=in_x,
            units=1,
            activation=self.ranking_activation,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name="nn_final"
        )

        self.nn_cls = tf.layers.dense(
            inputs=in_x,
            units=1,
            activation=self.ranking_activation,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name="nn_final",
            reuse=True
        )

        self.nn_group_0 = tf.gather(self.nn, self.y_fair_0)
        self.nn_group_1 = tf.gather(self.nn, self.y_fair_1)

        self.ranking_cost, self.fair_cost = self._def_cost(self.nn, self.y, self.nn_group_0, self.nn_group_1,
                                                           self.gamma)

    def listnet_loss(self, nn, y):
        """
        The Top-1 approximated ListNet loss as in Cao et al (2006), Learning to
        Rank: From Pairwise Approach to Listwise Approach
        :param nn: activation of the previous layer
        :param y: target labels
        :return: The loss
        """

        # ListNet top-1 reduces to a softmax and simple cross entropy
        st = tf.nn.softmax(y, axis=0)
        sx = tf.nn.softmax(nn, axis=0)
        return -tf.reduce_sum(st * tf.math.log(sx))

    def listnet_loss_fair(self, nn, nn_group_0, nn_group_1):
        """
        The Top-1 Deltr loss as in Meike Zehlike et al (2019),
        Reducing Disparate Exposure in Ranking: A Learning to Rank Approach
        :param nn: activation of the previous layer
        :param nn_group_0: non-protected group (eg. men)
        :param nn_group_1: protected group (eg. women)
        :return: The loss
        """
        sum_nn = tf.reduce_sum(tf.exp(nn))
        log_2 = tf.math.log(2.)
        n_g0 = tf.cast(tf.size(nn_group_0), tf.float32)
        n_g1 = tf.cast(tf.size(nn_group_1), tf.float32)

        exposure_group_0 = tf.reduce_sum(
            tf.exp(nn_group_0) / sum_nn / log_2) / n_g0
        exposure_group_1 = tf.reduce_sum(
            tf.exp(nn_group_1) / sum_nn / log_2) / n_g1

        return tf.math.maximum(0., exposure_group_0 - exposure_group_1) ** 2

    def _def_cost(self, nn, y, nn_group_0, nn_group_1, gamma=0.5):
        """
        The Top-1 Deltr loss as in Meike Zehlike et al (2019),
        Reducing Disparate Exposure in Ranking: A Learning to Rank Approach
        :param nn: activation of the previous layer
        :param y: target labels
        :param nn_group_0: non-protected group (eg. men)
        :param nn_group_1: protected group (eg. women)
        :param gamma: scaling factor of fair loss
        :return: The loss
        """
        list_loss = self.listnet_loss(nn, y)
        fair_loss = gamma * self.listnet_loss_fair(nn, nn_group_0, nn_group_1)
        return list_loss, fair_loss

    def _get_feed_dict(self, x, y, y_bias, samples):
        idx = np.random.randint(0, len(x), samples)
        x_i = x[idx]
        y_i = np.array([y[idx]]).transpose()
        if y_i.ndim > 2:  # if compas; not if synth
            y_i = y_i.reshape((y_i.shape[1], 1))
        y_fair_i = y_bias[idx][:, 0]
        group_0 = np.where(y_fair_i == 0)[0]  # non-protected group (eg. men)
        group_1 = np.where(y_fair_i == 1)[0]  # protected group (eg. women)

        return {self.x: x_i, self.y: y_i, self.y_fair_0: group_0, self.y_fair_1: group_1}

    def _get_feed_dict_queries(self, x, y, y_bias, samples, around=30):
        """ TODO: No queries here """
        idx = np.random.randint(0, len(x), samples)
        x_i = x[idx]
        y_i = np.array([y[idx]]).transpose()
        if y_i.ndim > 2:  # if compas; not if synth
            y_i = y_i.reshape((y_i.shape[1], 1))
        y_fair_i = y_bias[idx][:, 0]
        group_0 = np.where(y_fair_i == 0)[0]  # non-protected group (eg. men)
        group_1 = np.where(y_fair_i == 1)[0]  # protected group (eg. women)

        return {self.x: x_i, self.y: y_i, self.y_fair_0: group_0, self.y_fair_1: group_1}

    def fit(self, x, y, y_bias, **fit_params):
        """
        :param features: list of queries for training the net
        :param real_classes: list of labels inside a query
        :return:
        """
        if y_bias[0].shape[0] > 1:
            y_bias = y_bias[:, 0].reshape(-1, 1)

        x = np.array(x)
        y = np.array(y)
        y_bias = np.array(y_bias)

        super().fit(x, y, y_bias, **fit_params)

    def predict(self, features):
        """
        :param features: list of features of the instance feed to the net
        :return: predicted class
        """
        if len(features.shape) == 1:
            features = [features]

        res = self.sess.run(self.nn, feed_dict={self.x: features})

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
        return self.sess.run(self.nn, feed_dict={self.x: x0})
