import tensorflow as tf

if tf.__version__.split('.')[0] == '2':
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
from FairRanking.models.flip_gradient import flipGradient
from FairRanking.models.BaseDirectRanker import BaseDirectRanker, create_scalar_placeholder, create_placeholder


class directRankerAdvFlip(BaseDirectRanker):
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
                 fair_loss="ranking",
                 start_qids=10,
                 dataset=None,
                 random_seed=42,
                 name="DirectRankerAdvFlip",
                 gamma=1.,
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
        self.fair_loss = fair_loss

    def _build_model(self):
        """
        This function builds the directRanker with the values specified in the constructor
        :return:
        """

        self.x0 = create_placeholder(self.num_features)
        self.x1 = create_placeholder(self.num_features)
        self.y = create_placeholder(1)
        self.y_bias = create_placeholder(1)
        self.y_bias_0 = create_placeholder(self.num_fair_classes)
        self.y_bias_1 = create_placeholder(self.num_fair_classes)
        self.adj_loss_term = create_scalar_placeholder()

        if self.noise_module:
            in_0, in_1 = self.create_noise_module(self.x0, self.x1)
        else:
            in_0 = self.x0
            in_1 = self.x1

        for i, num_nodes in enumerate(self.hidden_layers):
            in_0, in_1 = self.create_feature_layers(in_0, in_1, num_nodes, 'nn_hidden_{}'.format(i))

        self.extracted_features = in_0

        # Create ranking part
        self.nn, self.nn_cls = self.create_ranking_layers(in_0, in_1, return_aux_ranker=True)

        # Create symflip layer for bias part
        nn_bias0 = flipGradient(in_0)
        nn_bias1 = flipGradient(in_1)

        self.nn_bias = (nn_bias0 - nn_bias1) / 2

        self.nn_bias = tf.layers.dense(
            inputs=self.nn_bias,
            units=1,
            activation=self.ranking_activation,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name="nn_bias"
        )

        # Create adv layer
        for i, num_nodes in enumerate(self.bias_layers):
            nn_bias0, nn_bias1 = self.create_debias_layers(nn_bias0, nn_bias1, self.feature_activation,
                                                           'nn_bias_{}'.format(i),
                                                           num_units=num_nodes)

        self.nn_bias_0 = nn_bias0
        self.nn_bias_1 = nn_bias1

        self.ranking_cost, self.fair_cost, self.fair_cost_extra = self._def_cost(
            self.nn, self.y, self.nn_bias, self.y_bias,
            self.nn_bias_0, self.y_bias_0,
            self.nn_bias_1, self.y_bias_1, self.gamma,
            self.adj_loss_term
        )

    def _def_cost(self, nn, y, nn_bias, y_bias, nn_bias_0, y_bias_0, nn_bias_1, y_bias_1, gamma, adj_loss_term):
        ranking_loss = (y - nn) ** 2
        fairness_loss_a = gamma * (
                tf.nn.softmax_cross_entropy_with_logits(labels=y_bias_0,
                                                        logits=nn_bias_0) \
                + tf.nn.softmax_cross_entropy_with_logits(labels=y_bias_1,
                                                          logits=nn_bias_1))
        if self.fair_loss == "ranking":
            fairness_loss_s = gamma * (y_bias - nn_bias) ** 2
        elif self.fair_loss == "abs":
            fairness_loss_s = gamma * tf.math.abs((tf.math.abs(y_bias) - tf.math.abs(nn_bias)))
        elif self.fair_loss == "adj":
            fairness_loss_s = gamma * tf.multiply(y_bias - nn_bias, tf.math.abs(y_bias)) * adj_loss_term
        else:
            raise ValueError('Loss called {} not recognized.'.format(self.fair_cost))

        return tf.reduce_mean(ranking_loss), \
               tf.reduce_mean(fairness_loss_a), \
               tf.reduce_mean(fairness_loss_s)

    def _get_feed_dict(self, x, y, y_bias, samples):
        pairs_dict = super()._get_feed_dict(x, y, y_bias, samples)

        return {self.x0: pairs_dict['x0'], self.x1: pairs_dict['x1'], self.y: pairs_dict['y_train'],
                self.y_bias: pairs_dict['y_bias'], self.y_bias_0: pairs_dict['y_bias_0'],
                self.y_bias_1: pairs_dict['y_bias_1'], self.adj_loss_term: pairs_dict['adj_loss_term']}

    def _get_feed_dict_queries(self, x, y, y_bias, samples, around=30):
        pairs_dict = super()._get_feed_dict_queries(x, y, y_bias, samples)

        return {self.x0: pairs_dict['x0'], self.x1: pairs_dict['x1'], self.y: pairs_dict['y_train'],
                self.y_bias: pairs_dict['y_bias_0']}
