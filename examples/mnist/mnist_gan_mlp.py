"""."""

import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from no_imagination.utils import get_current_timestamp, get_project_directory
from pylogging import HandlerType, setup_logger
from tensorflow.examples.tutorials.mnist import input_data

logger = logging.getLogger(__name__)


class GAN_MLP(object):
    """."""

    def __init__(self, z_dim=1):
        """."""
        logger.info("MNIST GAN MLP")
        logger.info("z_dim: {}".format(z_dim))

        mnist_data_dir = get_project_directory("mnist", "dataset")
        logger.info("mnist_data_dir: {}".format(mnist_data_dir))

        self.mnist_data = input_data.read_data_sets(mnist_data_dir, one_hot=True)
        self.z_dim = z_dim
        self.z_mean = np.zeros(self.z_dim)
        self.z_cov = np.diag(np.ones(self.z_dim))
        self.x_dim = 784
        self.y_dim = 10
        self.real_prob_val = 0.9
        self.fake_prob_val = 0.1
        self.__build_model()
        self.__start_session()

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

    def __fc_layer(self, scope_name, inputs, input_dim, output_dim, activation, keep_prob=None):
        """."""
        with tf.variable_scope(scope_name):
            w_fc = self.__weight_variable(shape=[input_dim, output_dim])
            b_fc = self.__bias_variable(shape=[output_dim])
            if activation == "linear":
                h_fc = tf.matmul(inputs, w_fc) + b_fc
            elif activation == "leaky_relu":
                h_fc = self.__leaky_relu(self.__batch_norm(tf.matmul(inputs, w_fc) + b_fc), alpha=0.2)
            else:
                h_fc = activation(tf.matmul(inputs, w_fc) + b_fc)

            if keep_prob is not None:
                h_fc = tf.nn.dropout(h_fc, keep_prob=keep_prob)

            return h_fc

    def __build_generator(self, inputs, input_dim, output_dim, keep_prob=None):
        """."""
        logger.info("Entry")
        fc1 = self.__fc_layer("fc1", inputs, input_dim, 256, activation="leaky_relu", keep_prob=keep_prob)
        fc2 = self.__fc_layer("fc2", fc1, 256, 512, activation="leaky_relu", keep_prob=keep_prob)
        fc3 = self.__fc_layer("fc3", fc2, 512, 1024, activation="leaky_relu", keep_prob=keep_prob)
        fc4 = self.__fc_layer("fc4", fc3, 1024, output_dim, activation=tf.nn.tanh)
        return fc4

    def __build_discriminator(self, inputs, input_dim, output_dim, keep_prob=None):
        """."""
        logger.info("Entry")
        fc1 = self.__fc_layer("fc1", inputs, input_dim, 1024, activation="leaky_relu", keep_prob=keep_prob)
        fc2 = self.__fc_layer("fc2", fc1, 1024, 512, activation="leaky_relu", keep_prob=keep_prob)
        fc3 = self.__fc_layer("fc3", fc2, 512, 256, activation="leaky_relu", keep_prob=keep_prob)
        fc4_logit = self.__fc_layer("fc4", fc3, 256, output_dim, activation="linear")
        return fc4_logit

    def __build_model(self):
        """."""
        logger.info("Entry")
        self.g = tf.Graph()
        with self.g.as_default():
            self.x = tf.placeholder(tf.float32, shape=[None, self.x_dim])
            self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])
            self.keep_prob = tf.placeholder(tf.float32)
            self.real_prob = tf.placeholder(tf.float32, shape=[None, 1])
            self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])

            with tf.variable_scope("G"):
                self.G = self.__build_generator(self.z, input_dim=self.z_dim, output_dim=self.x_dim)

            with tf.variable_scope("D") as scope:
                self.D1_logit = self.__build_discriminator(self.x, input_dim=self.x_dim,
                                                           output_dim=self.y_dim, keep_prob=self.keep_prob)
                self.D1 = tf.nn.sigmoid(self.D1_logit)
                scope.reuse_variables()
                self.D2_logit = self.__build_discriminator(self.G, input_dim=self.x_dim,
                                                           output_dim=self.y_dim, keep_prob=self.keep_prob)
                self.D2 = tf.nn.sigmoid(self.D2_logit)

            self.vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="G")
            self.vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="D")

            # self.loss_D_pre = tf.reduce_mean(tf.square(self.D1 - self.real_prob))
            # optimizer_D_pre = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
            # self.opt_D_pre = optimizer_D_pre.minimize(loss=self.loss_D_pre, var_list=self.vars_D)

            self.loss_D = (tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_logit,
                                                                   labels=tf.ones_like(self.D1_logit)) +
                           tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_logit,
                                                                   labels=tf.ones_like(self.D2_logit)))
            optimizer_D = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
            self.opt_D = optimizer_D.minimize(loss=self.loss_D, var_list=self.vars_D)
            self.accuracy_D = tf.reduce_sum(tf.cast(tf.equal(tf.round(self.D1_logit), tf.round(self.y)), tf.float32))

            self.loss_G = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_logit,
                                                                  labels=tf.ones_like(self.D2_logit))
            optimizer_G = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
            self.opt_G = optimizer_G.minimize(loss=self.loss_G, var_list=self.vars_G)

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

    def get_fake_batches(self, batch_size, random_seed=None):
        """."""
        batch_z = self.get_noise_batches(batch_size, random_seed)
        batch_fx, = self.sess.run([self.G], feed_dict={self.z: batch_z})  # comma for unpack list
        batch_fy = np.zeros(batch_size)
        return (batch_fx, batch_fy)

    def get_data_batches(self, batch_size, normalize):
        """."""
        if normalize == "normalize":
            batch_x, batch_y = self.mnist_data.train.next_batch(batch_size)
            batch_x = 2 * batch_x - 1e-4
            return batch_x, batch_y
        elif normalize == "standardize":
            raise NotImplementedError
        else:
            return self.mnist_data.train.next_batch(batch_size)

    def plot_losses(self, epoch, loss_G, loss_D):
        """."""
        pass

    def plot_fake_data(self, epoch, fig_plots=(10, 10), figsize=(10, 10), random_seed=None):
        """."""
        batch_fx, _ = self.get_fake_batches(batch_size=fig_plots[0] * fig_plots[1], random_seed=random_seed)
        batch_fx = batch_fx.reshape(-1, 28, 28)
        plt.figure(figsize=figsize)
        for i in range(batch_fx.shape[0]):
            plt.subplot(fig_plots[0], fig_plots[1], i + 1)
            plt.imshow(batch_fx[i], interpolation='nearest', cmap='gray_r')
            plt.axis('off')
        plt.tight_layout()
        img_dir = get_project_directory("mnist", "results", self.start_timestamp)
        plt.savefig(os.path.join(img_dir, "mnist_gan_mlp_{}.png".format(epoch)))
        plt.close()

    def pre_train(self, batch_size):
        """."""
        # Train on real mini batch
        batch_x, _ = self.get_data_batches(batch_size, normalize="normalize")
        batch_real_prob = np.ones([batch_size, 1]) * self.real_prob_val
        self.sess.run(self.opt_D_pre, feed_dict={self.x: batch_x, self.real_prob: batch_real_prob})

        # Train on fake mini batch
        batch_fx, _ = self.get_fake_batches(batch_size)
        batch_real_prob = np.ones([batch_size, 1]) * self.fake_prob_val
        self.sess.run(self.opt_D_pre, feed_dict={self.x: batch_fx, self.real_prob: batch_real_prob})

    def run(self, epochs, batch_size, summary_epochs=100):
        """."""
        self.start_timestamp = get_current_timestamp()
        start_time = time.time()

        for i in range(epochs):
            # pre train
            # if i == 0:
            #     logger.info("PRE-TRAINING")
            #     for j in range(60000 // batch_size):
            #         self.pre_train(batch_size)
            #
            #         if j % 50 == 0:
            #             # real behavior
            #             batch_x, _ = self.get_data_batches(batch_size, normalize="normalize")
            #             batch_real_prob = np.ones([batch_size, 1]) * self.real_prob_val
            #             real_loss_D_pre, real_D1 = self.sess.run([self.loss_D_pre, self.D1],
            #                                                      feed_dict={self.x: batch_x, self.real_prob: batch_real_prob})
            #             # fake behavior
            #             batch_fx, _ = self.get_fake_batches(batch_size)
            #             batch_real_prob = np.ones([batch_size, 1]) * self.fake_prob_val
            #             fake_loss_D_pre, fake_D1 = self.sess.run([self.loss_D_pre, self.D1],
            #                                                      feed_dict={self.x: batch_fx, self.real_prob: batch_real_prob})
            #
            #             time_diff = time.time() - start_time
            #             start_time = time.time()
            #
            #             logger.info("PreT-Batch: {:4d} - Real_D: {} - Real_Loss: {:0.4f} - Fake_D: {} - Fake_Loss: {:0.4f} - Time: {:0.1f}"
            #                         .format(j, real_D1[0], real_loss_D_pre, fake_D1[0], fake_loss_D_pre, time_diff))

            for j in range(60000 // batch_size):
                # Train discriminator
                batch_x, batch_y = self.get_data_batches(batch_size, normalize="No")
                batch_z = self.get_noise_batches(batch_size)
                self.sess.run([self.opt_D], feed_dict={self.x: batch_x, self.y: batch_y,
                                                       self.z: batch_z, self.keep_prob: 0.5})

                # Train generator
                batch_z = self.get_noise_batches(batch_size)
                self.sess.run([self.opt_G], feed_dict={self.z: batch_z, self.keep_prob: 1.0})

                if j % 100 == 0:
                    logger.info("Step: {:4d}".format(j))

            if i % summary_epochs == 0:
                batch_x, batch_y = self.get_data_batches(batch_size, normalize="No")
                batch_z = self.get_noise_batches(batch_size)
                loss_G, loss_D, D1, D2 = self.sess.run([self.loss_G, self.loss_D, self.D1, self.D2],
                                                       feed_dict={self.x: batch_x, self.y: batch_y, self.z: batch_z, self.keep_prob: 1.0})
                self.plot_losses(epoch=i, loss_G=loss_G, loss_D=loss_D)
                self.plot_fake_data(epoch=i, random_seed=1)
                time_diff = time.time() - start_time
                start_time = time.time()
                logger.info("Epoch: {:4d} \n- Loss_G: {} \n- Loss_D: {} \n- D1: - D2:  - Time: {:0.1f}"
                            .format(i, loss_G[0], loss_D[0], time_diff))


if __name__ == "__main__":
    setup_logger(log_directory=get_project_directory("mnist", "logs"),
                 file_handler_type=HandlerType.TIME_ROTATING_FILE_HANDLER,
                 allow_console_logging=True,
                 allow_file_logging=True,
                 max_file_size_bytes=10000,
                 change_log_level=None)
    mnist_gan_mlp = GAN_MLP(z_dim=10)
    mnist_gan_mlp.run(epochs=1000, batch_size=128, summary_epochs=1)
