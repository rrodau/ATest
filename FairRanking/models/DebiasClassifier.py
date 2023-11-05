import tensorflow as tf
import numpy as np

if tf.__version__.split('.')[0] == '2':
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()

from FairRanking.models.flip_gradient import flipGradient
from FairRanking.models.BaseDirectRanker import BaseDirectRanker, create_placeholder


class DebiasClassifier(BaseDirectRanker):

    def __init__(self,
                 hidden_layers=[10, 5],
                 bias_layers=[50, 20, 2],
                 feature_activation=tf.nn.sigmoid,
                 ranking_activation=tf.nn.sigmoid,
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
                 dataset=None,
                 name="DebiasClassifier",
                 gamma=1.,
                 noise_module=False,
                 noise_type='sigmoid_full',
                 whiteout=False,
                 uniform_noise=0,
                 whiteout_gamma=1.,
                 whiteout_lambda=1.,
                 num_features=0,
                 num_fair_classes=0,
                 save_dir=None,
                 num_relevance_classes=0):
        super().__init__(hidden_layers=hidden_layers,
                         feature_activation=feature_activation, ranking_activation=ranking_activation,
                         feature_bias=feature_bias, kernel_initializer=kernel_initializer,
                         start_batch_size=start_batch_size, end_batch_size=end_batch_size,
                         learning_rate=learning_rate, max_steps=max_steps, dataset=dataset,
                         learning_rate_step_size=learning_rate_step_size,
                         learning_rate_decay_factor=learning_rate_decay_factor, optimizer=optimizer,
                         print_step=print_step,
                         end_qids=end_qids, start_qids=start_qids, random_seed=random_seed, name=name, gamma=gamma,
                         noise_module=noise_module, noise_type=noise_type, whiteout=whiteout,
                         uniform_noise=uniform_noise,
                         whiteout_gamma=whiteout_gamma, whiteout_lambda=whiteout_lambda, num_features=num_features,
                         num_fair_classes=num_fair_classes, save_dir=save_dir)
        self.bias_layers = bias_layers
        self.num_relevance_classes = num_relevance_classes

    def _build_model(self):
        self.x0 = create_placeholder(self.num_features)
        self.y = create_placeholder(self.num_relevance_classes)
        self.y_bias = create_placeholder(self.num_fair_classes)

        nn = self.x0
        for i, num_neurons in enumerate(self.hidden_layers):
            nn = tf.layers.dense(
                inputs=nn,
                units=num_neurons,
                activation=tf.math.sigmoid,
                kernel_initializer=self.kernel_initializer,
                name="nn_fc_{}".format(i)
            )

        self.extracted_features = nn
        nn_bias = flipGradient(nn)

        for i, num_neurons in enumerate(self.bias_layers):
            nn_bias = tf.layers.dense(
                inputs=nn_bias,
                units=num_neurons,
                activation=tf.math.sigmoid if i != len(self.bias_layers) - 1 else None,
                kernel_initializer=self.kernel_initializer,
                name="nn_debias_{}".format(i)
            )

        for i, num_neurons in enumerate(self.bias_layers[:-1]):
            nn = tf.layers.dense(
                inputs=nn,
                units=num_neurons,
                activation=tf.math.sigmoid,
                kernel_initializer=self.kernel_initializer,
                name="nn_cls_{}".format(i)
            )

        nn = tf.layers.dense(
            inputs=nn,
            units=self.num_relevance_classes,
            activation=None,
            kernel_initializer=self.kernel_initializer,
            name="nn_cls_out"
        )

        self.nn_bias = nn_bias
        self.nn = nn
        self.nn_cls = nn

        self.ranking_cost, self.fair_cost = self._def_cost(self.nn, self.y, self.nn_bias, self.y_bias, self.gamma)

    def _def_cost(self, nn, y, nn_bias, y_bias, gamma):
        ranking_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=nn)
        fairness_loss = gamma * tf.nn.softmax_cross_entropy_with_logits(labels=y_bias, logits=nn_bias)
        return tf.reduce_mean(ranking_loss), \
               tf.reduce_mean(fairness_loss)

    def _get_feed_dict(self, x, y, y_bias, samples):
        idx = np.random.choice(len(x), samples)
        x_batch = x[idx]
        y_batch = one_hot_convert(y[idx], self.num_relevance_classes)
        y_bias_batch = y_bias[idx]
        return {self.x0: x_batch, self.y: y_batch, self.y_bias: y_bias_batch}


def one_hot_convert(y, num_classes):
    arr = np.zeros((len(y), num_classes))
    for i, yi in enumerate(y):
        arr[i, int(yi) - 1] = 1
    return arr
