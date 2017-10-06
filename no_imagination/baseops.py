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

    def __get_batch_norm_instance(self):
        """Return an instance of the batch_norm class that can be __called__."""
        # Taken as is from https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py
        class batch_norm(object):
            """Code modification of http://stackoverflow.com/a/33950177."""

            def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
                with tf.variable_scope(name):
                    self.epsilon = epsilon
                    self.momentum = momentum

                    self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
                    self.name = name

            def __call__(self, x, train=True):
                shape = x.get_shape().as_list()

                if train:
                    with tf.variable_scope(self.name):
                        self.beta = tf.get_variable("beta", [shape[-1]],
                                                    initializer=tf.constant_initializer(0.))
                        self.gamma = tf.get_variable("gamma", [shape[-1]],
                                                     initializer=tf.random_normal_initializer(1., 0.02))

                        try:
                            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                        except Exception:
                            batch_mean, batch_var = tf.nn.moments(x, [0, 1], name='moments')

                        ema_apply_op = self.ema.apply([batch_mean, batch_var])
                        self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                        with tf.control_dependencies([ema_apply_op]):
                            mean, var = tf.identity(batch_mean), tf.identity(batch_var)
                else:
                    mean, var = self.ema_mean, self.ema_var

                normed = tf.nn.batch_norm_with_global_normalization(
                    x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

                return normed

        return batch_norm()

    def relu_batch_norm(self, x):
        """."""
        return tf.nn.relu(self.__get_batch_norm_instance()(x=x))

    def leaky_relu_batch_norm(self, x):
        """."""
        return self.leaky_relu(self.__get_batch_norm_instance()(x=x))

    def __max_unpool(pool, ind, ksize=[1, 2, 2, 1], scope='unpool'):
        """.

        Source: https://github.com/tensorflow/tensorflow/issues/2169#issuecomment-315736497

        Unpooling layer after max_pool_with_argmax.

        Args:
           pool:   max pooled output tensor
           ind:      argmax indices
           ksize:     ksize is the same as for the pool

        Returns:
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
            if activation == "linear":
                h_fc = h_fc_logits
            elif activation == "leaky_relu":
                h_fc = self.leaky_relu(h_fc_logits, alpha=0.2)
            else:
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

    def deconv(self, inputs, filter_shape_hw, output_shape, activation, stride_hw=[1, 1],
               unpool=False, unpool_stride=[1, 2, 2, 1], scope=None, padding="SAME"):
        """Deconvolution operation.

        filter_shape = [filter_height, filter_width, output_channels, input_channels]
        The intuition for filter shape is that the deconvolution operation is a `one-to-many` operation:
        * There are as many operations as there are pixels in the input image.
        * Every operation is applied on one pixel to yield a filter_output of shape
            [filter_height, filter_width, output_channels] - which is basically the `one-to-many` operation.
        * Each pixel contains several `input_channels`.
        * For each operation (i.e. for each pixel) "n" sub_filter_outputs are generated which are summed up
            to form the filter_output, where "n" is equal to `input_channels`.
        * Therefore we need a filter of shape [filter_height, filter_width, output_channels, input_channels]
        * The filter_outputs from all operations are superimposed/merged together based on the `stride` and
            `padding` to eventually form the output image.

        Args:
            inputs (4d-tensor): Usually the output of a convolution or deconvolution operation. The shape of the
                of the inputs i.e. inputs.shape should be of the form [batch_size or ?, input_height, input_width, input_channels].
            output_shape (4d-list): The shape of the output which should be of the form [batch_size, output_height, output_width, output_channels].
            activation (op): ...
            stride (4d-list): ...

        Returns:
            4d-tensor: The deconvolution output after ACTIVATION.
        """
        logger.debug("[{}:deconv] Entry".format(scope))
        logger.debug("[{}:deconv] input_shape: {}".format(scope, inputs.get_shape().as_list()))
        logger.debug("[{}:deconv] output_shape: {}".format(scope, output_shape))
        assert isinstance(output_shape, list) and len(output_shape) == 4, "output_shape error: {}".format(output_shape)
        with tf.variable_scope(scope or "deconv"):
            # if unpool:
            #     inputs = self.__max_unpool(inputs, ksize=unpool_stride, strides=unpool_stride, padding="SAME")
            filter_shape = [filter_shape_hw[0], filter_shape_hw[1], output_shape[-1], inputs.get_shape().as_list()[-1]]
            logger.debug("[{}:deconv] filter_shape: {}".format(scope, filter_shape))

            W_deconv = self.__weight_variable(filter_shape)
            b_deconv = self.__bias_variable([output_shape[-1]])
            stride = [1, stride_hw[0], stride_hw[1], 1]
            deconv = tf.nn.conv2d_transpose(inputs, filter=W_deconv,
                                            output_shape=output_shape, strides=stride, padding=padding)
            deconv = activation(tf.nn.bias_add(deconv, b_deconv))
            logger.debug("[{}:deconv] deconv_shape: {}".format(scope, deconv.get_shape().as_list()))

            return deconv
