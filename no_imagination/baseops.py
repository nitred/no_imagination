"""."""
import logging

import tensorflow as tf

logger = logging.getLogger(__name__)


class BaseOps(object):
    """."""

    def __init__(self):
        """."""
        pass

    def __weight_variable(self, shape, initializer=tf.truncated_normal_initializer(stddev=0.1)):
        """."""
        return tf.get_variable("weights", shape=shape, dtype=tf.float32, initializer=initializer)

    def __bias_variable(self, shape, initializer=tf.constant_initializer(value=0.0)):
        """."""
        return tf.get_variable("biases", shape=shape, dtype=tf.float32, initializer=initializer)

    def leaky_relu(self, logits, alpha=0.2):
        """."""
        return tf.maximum(alpha * logits, logits, "leaky_relu")

    def linear(self, logits):
        """."""
        return logits

    def __batch_norm(self, x):
        return tf.contrib.layers.batch_norm(x, decay=0.9, scale=True, updates_collections=None)

    def __max_unpool(pool, ind, ksize=[1, 2, 2, 1], scope='unpool'):
        """.

           Unpooling layer after max_pool_with_argmax.
           Args:
               pool:   max pooled output tensor
               ind:      argmax indices
               ksize:     ksize is the same as for the pool
           Return:
               unpool:    unpooling tensor
        """
        with tf.variable_scope(scope):
            input_shape = tf.shape(pool)
            output_shape = [input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3]]

            flat_input_size = tf.cumprod(input_shape)[-1]
            flat_output_shape = tf.stack([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]])

            pool_ = tf.reshape(pool, tf.stack([flat_input_size]))
            batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                                     shape=tf.stack([input_shape[0], 1, 1, 1]))
            b = tf.ones_like(ind) * batch_range
            b = tf.reshape(b, tf.stack([flat_input_size, 1]))
            ind_ = tf.reshape(ind, tf.stack([flat_input_size, 1]))
            ind_ = tf.concat([b, ind_], 1)

            ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
            ret = tf.reshape(ret, tf.stack(output_shape))
            return ret

    def fc(self, inputs, output_dim, activation, keep_prob=None, scope=None):
        """.

        Args:
            inputs (2d-tensor): Inputs shape is of the form [batch_size, fc_dim]
        """
        logger.debug("[{}:fc] Entry".format(scope))
        logger.debug("[{}:fc] input_shape: {}".format(scope, inputs.get_shape().as_list()))
        logger.debug("[{}:fc] output_dim: {}".format(scope, output_dim))
        with tf.variable_scope(scope or "fc"):
            input_dim = inputs.get_shape().as_list()[-1]
            w_fc = self.__weight_variable(shape=[input_dim, output_dim])
            b_fc = self.__bias_variable(shape=[output_dim])
            h_fc_logits = tf.matmul(inputs, w_fc) + b_fc
            h_fc = activation(h_fc_logits)

            if keep_prob is not None:
                h_fc = tf.nn.dropout(h_fc, keep_prob=keep_prob)

            logger.debug("[{}:fc] fc_shape: {}".format(scope, h_fc.get_shape().as_list()))
            return h_fc

    def conv(self, inputs, filter_shape, activation, stride=[1, 1, 1, 1], pool=False, pool_stride=[1, 2, 2, 1], scope=None):
        """.

        Filter Arguments Example:
        >>> filter_shape = [5, 5, 1, 32]  # 5x5 filter and 1 channel, 32 filters or feature maps
        """
        with tf.variable_scope(scope or "conv"):
            W_conv = self.__weight_variable(filter_shape)
            b_conv = self.__bias_variable([filter_shape[-1]])
            output = tf.nn.conv2d(input=inputs, filter=W_conv, strides=stride, padding="SAME")
            output = activation(output + b_conv)

            if pool:
                output = tf.nn.max_pool(output, ksize=pool_stride, strides=pool_stride, padding="SAME")

            return output

    def deconv(self):
        """."""
        with tf.variable_scope("deconv"):
            pass
