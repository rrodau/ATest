from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

if tf.__version__.split('.')[0] == '2':
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()

from tensorflow.python.framework import ops


class FlipGradientBuilder(object):
    """
    Taken from https://github.com/pumpikano/tf-dann/blob/master/flip_gradient.py
    """

    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1

        return y


flipGradient = FlipGradientBuilder()
