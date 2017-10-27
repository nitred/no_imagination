"""."""

import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from no_imagination.baseops import BaseOps
from no_imagination.utils import get_current_timestamp, get_project_directory
from pylogging import HandlerType, setup_logger
from tensorflow.contrib.layers import flatten
from tensorflow.examples.tutorials.mnist import input_data

logger = logging.getLogger(__name__)


class GAN_CNN(BaseOps):
    """."""

    def __init__(self, z_dim, batch_size):
        """."""
        logger.info("MNIST GAN CNN")
        logger.info("z_dim: {}".format(z_dim))

        mnist_data_dir = get_project_directory("mnist", "dataset")
        logger.info("mnist_data_dir: {}".format(mnist_data_dir))

        self.mnist_data = input_data.read_data_sets(mnist_data_dir, one_hot=True)
        self.z_dim = z_dim
        self.z_mean = np.zeros(self.z_dim)
        self.z_cov = np.diag(np.ones(self.z_dim))
        self.x_dim = 784
        self.y_dim = 10
        self.batch_size = batch_size
        self.G_input_dim = self.y_dim
        self.D_input_dim = self.x_dim + self.y_dim
        self.real_prob_val = 0.9
        self.fake_prob_val = 0.1
        self.plot_batch_x, self.plot_batch_y = self.get_data_batches(10 * 10, normalize="no")
        self.plot_batch_z = self.get_noise_batches(10 * 10, random_seed=1)
        self.std_test_data = False
        self.__build_model()
        self.__start_session()

    def __build_generator(self, inputs, input_dim, output_dim, keep_prob=None):
        """."""
        logger.info("Entry")
        fc1 = self.fc(inputs=inputs, output_dim=(7 * 7 * 128),
                      activation=self.linear, scope='fc1')
        fc1_folded = tf.reshape(fc1, [-1, 7, 7, 128])
        deconv1 = self.deconv(inputs=fc1_folded, filter_shape_hw=[5, 5], stride_hw=[2, 2],
                              output_shape=[self.batch_size, 14, 14, 64], activation=self.relu_batch_norm, scope='deconv1')

        # deconv2 = self.deconv(inputs=deconv1, filter_shape_hw=[5, 5], stride_hw=[2, 2],
        #                       output_shape=[self.batch_size, 28, 28, 32], activation=self.linear, scope='deconv2')
        deconv3 = self.deconv(inputs=deconv1, filter_shape_hw=[5, 5], stride_hw=[2, 2],
                              output_shape=[self.batch_size, 28, 28, 1], activation=tf.nn.tanh, scope='deconv3')
        # deconv3_flat = tf.reshape(deconv3, shape=[-1, output_dim])
        # deconv3_flat = tf.nn.tanh(deconv3_flat, name="deconv3_flat_tanh")
        return deconv3

        # fc1 = self.fc(inputs=inputs, output_dim=(14 * 14 * 1),
        #               activation=self.linear, scope='fc1')
        # fc1_folded = tf.reshape(fc1, [-1, 14, 14, 1])
        # deconv1 = self.deconv(inputs=fc1_folded, filter_shape_hw=[4, 4], stride_hw=[2, 2], output_shape=[self.batch_size, 28, 28, 1],
        #                       activation=tf.nn.tanh, scope='deconv1')
        # deconv1_flat = tf.reshape(deconv1, shape=[-1, 28 * 28])
        # # deconv1_flat = tf.nn.tanh(deconv1_flat, name="deconv1_flat_tanh")
        # deconv1_flat = tf.identity(deconv1_flat, name='deconv1_flat')
        #
        # # fc2 = self.fc(inputs=inputs, output_dim=output_dim,
        # #               activation=self.linear, scope='fc2')
        # return deconv1_flat

    def __build_discriminator(self, inputs, conditioning_vector, output_dim, keep_prob=None):
        """."""
        logger.info("Entry")
        conv1 = self.conv(inputs=inputs, filter_shape=[5, 5, 1, 16],
                          pool=False, activation=self.leaky_relu, stride=[1, 2, 2, 1], scope='conv1')
        conv2 = self.conv(inputs=conv1, filter_shape=[5, 5, 16, 32],
                          pool=False, activation=self.leaky_relu, stride=[1, 2, 2, 1], scope='conv2')

        conv2_concat = self.concat_conv_vec_and_cond_vec(conv_vec=conv2, cond_vec=conditioning_vector)

        conv3 = self.conv(inputs=conv2_concat, filter_shape=[5, 5, 32 + 10, 32],
                          pool=False, activation=self.leaky_relu, stride=[1, 1, 1, 1], scope='conv3')

        conv3_flat = flatten(conv3)

        fc4 = self.fc(inputs=conv3_flat, output_dim=output_dim, activation=tf.nn.sigmoid, scope='fc4')
        return fc4

    def __build_model(self):
        """."""
        logger.info("Entry")
        self.g = tf.Graph()
        with self.g.as_default():
            self.x = tf.placeholder(tf.float32, shape=[None, self.x_dim], name='x')
            self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim], name='y')
            self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.real_prob = tf.placeholder(tf.float32, shape=[None, 1], name='real_prob')

            # Need y, z, keep_prob
            with tf.variable_scope("G"):
                self.G_input = tf.concat(values=[self.z, self.y], axis=1)
                self.G = self.__build_generator(self.G_input,
                                                input_dim=self.G_input_dim,
                                                output_dim=self.x_dim)

            # Need x, y, z, keep_prob
            with tf.variable_scope("D") as scope:
                # Concat reduced_text_embedding to conv4.
                x_reshape = tf.reshape(self.x, shape=[-1, 28, 28, 1])
                # self.D1_input = self.concat_conv_vec_and_cond_vec(conv_vec=x_reshape, cond_vec=self.y)
                # self.D1_input = tf.concat(values=[self.x, self.y], axis=1)
                self.D1 = self.__build_discriminator(x_reshape,
                                                     conditioning_vector=self.y,
                                                     output_dim=self.y_dim,
                                                     keep_prob=self.keep_prob)
                self.D1 = tf.clip_by_value(self.D1, clip_value_min=1e-6, clip_value_max=1 - 1e-6)

                scope.reuse_variables()

                # self.D2_input = self.concat_conv_vec_and_cond_vec(conv_vec=self.G, cond_vec=self.y)
                # self.D2_input = tf.concat(values=[self.G, self.y], axis=1)
                self.D2 = self.__build_discriminator(self.G,
                                                     conditioning_vector=self.y,
                                                     output_dim=self.y_dim,
                                                     keep_prob=self.keep_prob)
                self.D2 = tf.clip_by_value(self.D2, clip_value_min=1e-6, clip_value_max=1 - 1e-6)

            # Trainable variables
            self.vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="G")
            self.vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="D")

            # Losses
            # self.loss_D_pre = tf.reduce_mean(tf.square(self.D1 - self.real_prob))
            self.loss_G = tf.reduce_mean(-tf.log(self.D2))
            self.loss_D = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))

            # Optimizers
            # optimizer_D_pre = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
            optimizer_G = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
            optimizer_D = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)

            # Optimize
            # self.opt_D_pre = optimizer_D_pre.minimize(loss=self.loss_D_pre, var_list=self.vars_D)
            self.opt_G = optimizer_G.minimize(loss=self.loss_G, var_list=self.vars_G)
            self.opt_D = optimizer_D.minimize(loss=self.loss_D, var_list=self.vars_D)

    def __start_session(self):
        """."""
        with self.g.as_default():
            self.sess = tf.Session(graph=self.g)
            self.sess.run(tf.global_variables_initializer())

    def get_noise_batches(self, batch_size, random_seed=None):
        """."""
        if random_seed:
            np.random.seed(random_seed)
        # batch_z = np.random.multivariate_normal(mean=self.z_mean, cov=self.z_cov, size=batch_size)
        batch_z = np.random.normal(size=[batch_size, self.z_dim])
        return batch_z

    def get_fake_batches(self, batch_size, keep_prob=1.0):
        """."""
        batch_fx, = self.sess.run([self.G], feed_dict={self.y: self.plot_batch_y,
                                                       self.z: self.plot_batch_z,
                                                       self.keep_prob: keep_prob})
        return batch_fx

    def get_data_batches(self, batch_size, normalize):
        """."""
        if normalize == "normalize":
            batch_x, batch_y = self.mnist_data.train.next_batch(batch_size)
            batch_x = 2 * batch_x - 1.0
            return batch_x, batch_y
        elif normalize == "standardize":
            raise NotImplementedError
        else:
            return self.mnist_data.train.next_batch(batch_size)

    def plot_losses(self, epoch, loss_G, loss_D):
        """."""
        pass

    def plot_fake_data(self, epoch, batch_fx, fig_plots=(10, 10), figsize=(10, 10)):
        """."""
        batch_fx = batch_fx.reshape(-1, 28, 28)
        plt.figure(figsize=figsize)
        for i in range(batch_fx.shape[0]):
            plt.subplot(fig_plots[0], fig_plots[1], i + 1)
            plt.imshow(batch_fx[i], interpolation='nearest', cmap='gray_r')
            plt.axis('off')
        plt.tight_layout()
        img_dir = get_project_directory("mnist_cnn_2", "results", self.start_timestamp)
        plt.savefig(os.path.join(img_dir, "mnist_gan_cnn_{}.png".format(epoch)))
        plt.close()

    def pre_train(self, batch_size):
        """."""
        # Train on real mini batch
        batch_x, _ = self.get_data_batches(batch_size, normalize="normalize")
        batch_real_prob = np.ones([batch_size, 1]) * self.real_prob_val
        self.sess.run(self.opt_D_pre, feed_dict={self.x: batch_x, self.real_prob: batch_real_prob})

        # Train on fake mini batch
        batch_fx = self.get_fake_batches(batch_size)
        batch_real_prob = np.ones([batch_size, 1]) * self.fake_prob_val
        self.sess.run(self.opt_D_pre, feed_dict={self.x: batch_fx, self.real_prob: batch_real_prob})

    def run(self, epochs, batch_size, summary_epochs=1):
        """."""
        self.batch_size = batch_size
        self.start_timestamp = get_current_timestamp()
        start_time = time.time()

        for i in range(epochs):
            for j in range(60000 // batch_size):
                # Train discriminator
                batch_x, batch_y = self.get_data_batches(batch_size, normalize="normalize")
                batch_z = self.get_noise_batches(batch_size)
                self.sess.run([self.opt_D], feed_dict={self.x: batch_x, self.y: batch_y,
                                                       self.z: batch_z, self.keep_prob: 0.5})

                # Train generator
                batch_x, batch_y = self.get_data_batches(batch_size, normalize="normalize")
                batch_z = self.get_noise_batches(batch_size)
                self.sess.run([self.opt_G], feed_dict={self.z: batch_z,  self.y: batch_y, self.keep_prob: 0.5})

                if j % 100 == 0:
                    # logger.info("Step: {:4d}".format(j))
                    pass

            if i % summary_epochs == 0:
                if not self.std_test_data:
                    self.std_batch_x, self.std_batch_y = self.get_data_batches(batch_size=10 * 10,
                                                                               normalize="normalize")
                    self.std_batch_z = self.get_noise_batches(batch_size=10 * 10)
                    self.std_test_data = True

                loss_G, loss_D, D1, D2, batch_fx = self.sess.run([self.loss_G, self.loss_D, self.D1, self.D2, self.G],
                                                                 feed_dict={self.x: self.std_batch_x, self.y: self.std_batch_y,
                                                                            self.z: self.std_batch_z, self.keep_prob: 1.0})
                self.plot_losses(epoch=i, loss_G=loss_G, loss_D=loss_D)
                self.plot_fake_data(epoch=i, batch_fx=batch_fx)
                time_diff = time.time() - start_time
                start_time = time.time()
                logger.info("Epoch: {:3d} - L_G: {:0.3f} - L_D: {:0.3f} - D1: {:0.3f} - D2: {:0.3f} - Time: {:0.1f}"
                            .format(i, loss_G, loss_D, D1[0][0], D2[0][0], time_diff))


if __name__ == "__main__":
    setup_logger(log_directory=get_project_directory("mnist_cnn_2", "logs"),
                 file_handler_type=HandlerType.TIME_ROTATING_FILE_HANDLER,
                 allow_console_logging=True,
                 allow_file_logging=True,
                 max_file_size_bytes=10000,
                 change_log_level=None)
    mnist_gan_mlp = GAN_CNN(z_dim=10, batch_size=100)
    mnist_gan_mlp.run(epochs=1000, batch_size=100, summary_epochs=1)
