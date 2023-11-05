import tensorflow as tf

if tf.__version__.split('.')[0] == '2':
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
from FairRanking.models.flip_gradient import flipGradient
from FairRanking.models.DirectRanker import directRanker
from FairRanking.models.BaseDirectRanker import BaseDirectRanker, create_placeholder, create_scalar_placeholder

import pickle


class directRankerAdv(BaseDirectRanker):
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
                 bias_layers=[50, 20, 2],
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
                 name="DirectRankerAdv",
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
        self.bias_layers = bias_layers

    def _build_model(self):
        """
        This function builds the ADV. directRanker with the values specified in the constructor
        """
        self.x0 = create_placeholder(self.num_features)
        self.x1 = create_placeholder(self.num_features)
        self.y = create_placeholder(1)
        self.y_bias_0 = create_placeholder(self.num_fair_classes)
        self.y_bias_1 = create_placeholder(self.num_fair_classes)

        if self.noise_module:
            in_0, in_1 = self.create_noise_module(self.x0, self.x1)
        else:
            in_0 = self.x0
            in_1 = self.x1

        for i, num_nodes in enumerate(self.hidden_layers):
            in_0, in_1 = self.create_feature_layers(in_0, in_1, num_nodes, 'nn_hidden_{}'.format(i))

        self.extracted_features = in_0

        nn_bias0 = flipGradient(in_0)
        nn_bias1 = flipGradient(in_1)

        for i, num_nodes in enumerate(self.bias_layers):
            nn_bias0, nn_bias1 = self.create_debias_layers(nn_bias0, nn_bias1, self.feature_activation,
                                                           'nn_bias_{}'.format(i),
                                                           num_units=num_nodes)

        self.nn_bias0 = nn_bias0
        self.nn_bias1 = nn_bias1

        self.nn, self.nn_cls = self.create_ranking_layers(in_0, in_1, return_aux_ranker=True)

        self.ranking_cost, self.fair_cost = self._def_cost(self.nn, self.y, self.nn_bias0, self.y_bias_0, self.nn_bias1,
                                                           self.y_bias_1, self.gamma)

    def _def_cost(self, nn, y0, nn_bias0, y0_bias0, nn_bias1, y0_bias1,
                  gamma):
        """
        Cost of the ADV. directRanker
        """
        ranking_loss = (y0 - nn) ** 2
        fairness_loss = gamma * (
                tf.nn.softmax_cross_entropy_with_logits(labels=y0_bias0,
                                                        logits=nn_bias0) \
                + tf.nn.softmax_cross_entropy_with_logits(labels=y0_bias1,
                                                          logits=nn_bias1))

        return tf.reduce_mean(ranking_loss), \
               tf.reduce_mean(fairness_loss)

    def _get_feed_dict(self, x, y, y_bias, samples):
        pairs_dict = super()._get_feed_dict(x, y, y_bias, samples)

        return {self.x0: pairs_dict['x0'], self.x1: pairs_dict['x1'], self.y: pairs_dict['y_train'],
                self.y_bias_0: pairs_dict['y_bias_0'], self.y_bias_1: pairs_dict['y_bias_1']}

    def _get_feed_dict_queries(self, x, y, y_bias, samples, around=30):
        pairs_dict = super()._get_feed_dict_queries(x, y, y_bias, samples)

        return {self.x0: pairs_dict['x0'], self.x1: pairs_dict['x1'],
                self.y_bias_0: pairs_dict['y_bias_0'], self.y_bias_1: pairs_dict['y_bias_1']}

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
