"""."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from no_imagination.utils import get_project_directory
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

    def __fc_layer(self, scope_name, inputs, input_dim, output_dim, activation, keep_prob=None):
        """."""
        with tf.variable_scope(scope_name):
            w_fc = self.__weight_variable(shape=[input_dim, output_dim])
            b_fc = self.__bias_variable(shape=[output_dim])
            if activation == "linear":
                h_fc = tf.matmul(inputs, w_fc) + b_fc
            elif activation == "leaky_relu":
                h_fc = self.__leaky_relu(tf.matmul(inputs, w_fc) + b_fc, alpha=0.2)
            else:
                h_fc = activation(tf.matmul(inputs, w_fc) + b_fc)

            if keep_prob:
                h_fc = tf.nn.dropout(h_fc, keep_prob=keep_prob)

            return h_fc

    def __build_generator(self, inputs, input_dim, output_dim):
        """."""
        logger.info("Entry")
        fc1 = self.__fc_layer("fc1", inputs, input_dim, 256, activation="leaky_relu")
        fc2 = self.__fc_layer("fc2", fc1, 256, 512, activation="leaky_relu")
        fc3 = self.__fc_layer("fc3", fc2, 512, 1024, activation="leaky_relu")
        fc4 = self.__fc_layer("fc4", fc3, 1024, output_dim, activation=tf.nn.tanh)
        return fc4

    def __build_discriminator(self, inputs, input_dim, output_dim):
        """."""
        logger.info("Entry")
        fc1 = self.__fc_layer("fc1", inputs, input_dim, 1024, activation="leaky_relu", keep_prob=1.0)
        fc2 = self.__fc_layer("fc2", fc1, 1024, 512, activation="leaky_relu", keep_prob=1.0)
        fc3 = self.__fc_layer("fc3", fc2, 512, 256, activation="leaky_relu", keep_prob=1.0)
        fc4 = self.__fc_layer("fc4", fc3, 256, output_dim, activation=tf.nn.sigmoid)
        return fc4

    def __build_model(self):
        """."""
        logger.info("Entry")
        self.g = tf.Graph()
        with self.g.as_default():
            self.x = tf.placeholder(tf.float32, shape=[None, self.x_dim])
            self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])

            with tf.variable_scope("G"):
                self.G = self.__build_generator(self.z, input_dim=self.z_dim, output_dim=self.x_dim)

            with tf.variable_scope("D") as scope:
                self.D1 = self.__build_discriminator(self.x, input_dim=self.x_dim, output_dim=1)
                scope.reuse_variables()
                self.D2 = self.__build_discriminator(self.G, input_dim=self.x_dim, output_dim=1)

            self.loss_G = tf.reduce_mean(-tf.log(self.D2))
            self.loss_D = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))

            self.vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="G")
            self.vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="D")

            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

            self.opt_G = optimizer.minimize(loss=self.loss_G, var_list=self.vars_G)
            self.opt_D = optimizer.minimize(loss=self.loss_D, var_list=self.vars_D)

    def __start_session(self):
        """."""
        with self.g.as_default():
            self.sess = tf.Session(graph=self.g)
            self.sess.run(tf.global_variables_initializer())

    def get_data_batches(self, batch_size):
        """."""
        return self.mnist_data.train.next_batch(batch_size)

    def get_noise_batches(self, batch_size, random_seed=None):
        """."""
        if random_seed:
            np.random.seed(random_seed)
        batch_z = np.random.multivariate_normal(mean=self.z_mean, cov=self.z_cov, size=batch_size)
        return batch_z

    def get_fake_batches(self, batch_size, random_seed=None):
        """."""
        batch_z = self.get_noise_batches(batch_size, random_seed)
        batch_fx, = self.sess.run([self.G], feed_dict={self.z: batch_z})  # comma for unpack list
        batch_fy = np.zeros(batch_size)
        return (batch_fx, batch_fy)

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
        img_dir = get_project_directory("mnist", "results")
        plt.savefig(os.path.join(img_dir, "mnist_gan_mlp_{}.png".format(epoch)))

    def run(self, epochs, batch_size, summary_epochs=100):
        """."""
        for i in range(epochs):
            for j in range(60000 // batch_size):
                # Train discriminator
                batch_x, _ = self.get_data_batches(batch_size)
                batch_z = self.get_noise_batches(batch_size)
                self.sess.run([self.opt_D], feed_dict={self.x: batch_x, self.z: batch_z})

                if j == 0:
                    # Train generator
                    batch_z = self.get_noise_batches(batch_size)
                    self.sess.run([self.opt_G], feed_dict={self.x: batch_x, self.z: batch_z})

            if i % summary_epochs == 0:
                batch_x, _ = self.get_data_batches(batch_size)
                batch_z = self.get_noise_batches(batch_size)
                loss_G, loss_D, D1, D2 = self.sess.run([self.loss_G, self.loss_D, self.D1, self.D2],
                                                       feed_dict={self.x: batch_x, self.z: batch_z})
                self.plot_losses(epoch=i, loss_G=loss_G, loss_D=loss_D)
                self.plot_fake_data(epoch=i, random_seed=1)
                logger.info(
                    "Epoch: {:4d} --- Loss_G: {:0.4f} --- Loss_D: {:0.4f} -- D1: {} --- D2: {}".format(i, loss_G, loss_D, D1, D2))


if __name__ == "__main__":
    setup_logger(log_directory=get_project_directory("mnist", "logs"),
                 file_handler_type=HandlerType.TIME_ROTATING_FILE_HANDLER,
                 allow_console_logging=True,
                 allow_file_logging=True,
                 max_file_size_bytes=100000,
                 change_log_level=None)
    mnist_gan_mlp = GAN_MLP(z_dim=10)
    mnist_gan_mlp.run(epochs=10, batch_size=128, summary_epochs=1)
