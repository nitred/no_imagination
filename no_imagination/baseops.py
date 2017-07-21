"""."""
import tensorflow as tf


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

    def __leaky_relu(self, inputs, alpha):
        """."""
        return tf.maximum(alpha * inputs, inputs, "leaky_relu")

    def __batch_norm(self, x):
        return tf.contrib.layers.batch_norm(x, decay=0.9, scale=True, updates_collections=None)

    def fc(self, inputs, output_dim, activation, keep_prob=None):
        """."""
        with tf.variable_scope("fc"):
            input_dim = inputs.shape[-1]
            w_fc = self.__weight_variable(shape=[input_dim, output_dim])
            b_fc = self.__bias_variable(shape=[output_dim])
            h_fc_logit = tf.matmul(inputs, w_fc) + b_fc

            if activation == "linear":
                h_fc = h_fc_logit
            elif activation == "leaky_relu":
                h_fc = self.__leaky_relu(h_fc_logit, alpha=0.2)
            else:
                h_fc = activation(h_fc_logit)

            if keep_prob is not None:
                h_fc = tf.nn.dropout(h_fc, keep_prob=keep_prob)

            return h_fc

    def conv(self, inputs, filter_shape, activation, stride=[1, 1, 1, 1], pool=True, pool_stride=[1, 2, 2, 1]):
        """.

        Filter Arguments Example:
        >>> filter_shape = [5, 5, 1, 32]  # 5x5 filter and 1 channel, 32 filters or feature maps
        """
        with tf.variable_scope("conv"):
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
