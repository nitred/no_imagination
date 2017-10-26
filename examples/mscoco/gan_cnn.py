"""GAN CNN Architecture.

Adapted from https://github.com/paarthneekhara/text-to-image/blob/master/model.py
Adapted from https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
"""

import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from no_imagination.baseops import BaseOps
from no_imagination.datasets.mscoco import mscoco_generator
from no_imagination.utils import get_current_timestamp, get_project_directory
from pylogging import HandlerType, setup_logger
from tensorflow.contrib.layers import flatten

logger = logging.getLogger(__name__)


class GAN_CNN(BaseOps):
    """."""

    def __init__(self, z_dim, batch_size):
        """."""
        logger.info("MSCOCO GAN CNN")
        logger.info("z_dim: {}".format(z_dim))

        # self.mnist_data = input_data.read_data_sets(mnist_data_dir, one_hot=True)
        self.z_dim = z_dim
        # self.z_mean = np.zeros(self.z_dim)
        # self.z_cov = np.diag(np.ones(self.z_dim))
        # self.x_dim = 784
        self.height = 128
        self.width = 128
        self.n_channels = 3
        self.captions_dim = 4800
        self.reduced_text_embedding_dim = 256
        self.d_first_filter_dim = 16
        self.g_first_filter_dim = 16
        # self.y_dim = 10
        self.batch_size = batch_size
        # self.G_input_dim = self.y_dim
        # self.D_input_dim = self.x_dim + self.y_dim
        # self.real_prob_val = 0.9
        # self.fake_prob_val = 0.1
        # self.plot_batch_x, self.plot_batch_y = self.get_data_batches(10 * 10, normalize="no")
        # self.plot_batch_z = self.get_noise_batches(10 * 10, random_seed=1)
        self.std_test_data = False
        self.__build_model()
        self.__start_session()

    def __build_generator(self, z, text_embedding, keep_prob=None):
        """."""
        logger.info("Entry")

        # Convert a possibly huge text_embedding into a more manageable reduced_text_embedding vector.
        reduced_text_embedding = self.fc(inputs=text_embedding, output_dim=self.reduced_text_embedding_dim,
                                         activation=self.leaky_relu, scope='reduced_text_embedding')

        # Concat the z_vector and reduced_text_embedding to form the input to the generator.
        g_input = tf.concat(values=[z, reduced_text_embedding], axis=1)

        # Expected Output Shape
        output_h, output_w, output_channels = self.height, self.width, self.n_channels  # 224/112
        output_h_by_2, output_w_by_2, g_channels_into_1 = self.height // 2, self.width // 2, self.g_first_filter_dim * 1  # 112/56, 64/16
        output_h_by_4, output_w_by_4, g_channels_into_2 = self.height // 4, self.width // 4, self.g_first_filter_dim * 2  # 56/28, 128/32
        output_h_by_8, output_w_by_8, g_channels_into_4 = self.height // 8, self.width // 8, self.g_first_filter_dim * 4  # 28/14, 256/64
        output_h_by_16, output_w_by_16, g_channels_into_8 = self.height // 16, self.width // 16, self.g_first_filter_dim * 8  # 14/7, 512/128

        # Project g_input using an FC so that it matches the dimensions required by deconv1.
        fc1 = self.fc(inputs=g_input, output_dim=output_h_by_16 * output_w_by_16 * g_channels_into_8,
                      activation=self.linear, scope='fc1')
        fc1_folded = tf.reshape(fc1, [-1, output_h_by_16, output_w_by_16, g_channels_into_8])

        # Deconv away!
        deconv1 = self.deconv(inputs=fc1_folded, filter_shape_hw=[5, 5], stride_hw=[2, 2], activation=self.relu_batch_norm,
                              output_shape=[self.batch_size, output_h_by_8, output_w_by_8, g_channels_into_4], scope='deconv1')
        deconv2 = self.deconv(inputs=deconv1, filter_shape_hw=[5, 5], stride_hw=[2, 2], activation=self.relu_batch_norm,
                              output_shape=[self.batch_size, output_h_by_4, output_w_by_4, g_channels_into_2], scope='deconv2')
        deconv3 = self.deconv(inputs=deconv2, filter_shape_hw=[5, 5], stride_hw=[2, 2], activation=self.relu_batch_norm,
                              output_shape=[self.batch_size, output_h_by_2, output_w_by_2, g_channels_into_1], scope='deconv3')
        deconv4 = self.deconv(inputs=deconv3, filter_shape_hw=[5, 5], stride_hw=[2, 2], activation=tf.nn.tanh,
                              output_shape=[self.batch_size, output_h, output_w, output_channels], scope='deconv4')

        return (deconv4 / 2.0) + 0.5

    def __build_discriminator(self, images, text_embeddings, keep_prob=None):
        """."""
        logger.info("Entry")

        # Precalculate the conv layer filter dimentions.
        conv1_filter_dim = self.d_first_filter_dim
        conv2_filter_dim = self.d_first_filter_dim * 2
        conv3_filter_dim = self.d_first_filter_dim * 4
        conv4_filter_dim = self.d_first_filter_dim * 8
        # Last two layers are the same.
        conv5_filter_dim = conv4_filter_dim

        # Taken from https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
        # NOTE: No pooling. Stride 2.
        conv1 = self.conv(inputs=images, filter_shape=[5, 5, self.n_channels, conv1_filter_dim],
                          pool=False, activation=self.leaky_relu, stride=[1, 2, 2, 1], scope='conv1')
        # conv1 output shape [-1, h/2, w/2, conv1_filter_dim]
        conv2 = self.conv(inputs=conv1, filter_shape=[5, 5, conv1_filter_dim, conv2_filter_dim],
                          pool=False, activation=self.leaky_relu_batch_norm, stride=[1, 2, 2, 1], scope='conv2')
        # conv2 output shape [-1, h/4, w/4, conv2_filter_dim]
        conv3 = self.conv(inputs=conv2, filter_shape=[5, 5, conv2_filter_dim, conv3_filter_dim],
                          pool=False, activation=self.leaky_relu_batch_norm, stride=[1, 2, 2, 1], scope='conv3')
        # conv3 output shape [-1, h/8, w/8, conv3_filter_dim]
        conv4 = self.conv(inputs=conv3, filter_shape=[5, 5, conv3_filter_dim, conv4_filter_dim],
                          pool=False, activation=self.leaky_relu_batch_norm, stride=[1, 2, 2, 1], scope='conv4')
        # conv4 output shape [-1, h/16, w/16, conv4_filter_dim]

        # Taken from https://github.com/paarthneekhara/text-to-image/blob/master/model.py
        # ADD TEXT EMBEDDING TO THE NETWORK
        # Convert a possibly huge text_embedding into a more manageable reduced_text_embedding vector.
        reduced_text_embeddings = self.fc(inputs=text_embeddings, output_dim=self.reduced_text_embedding_dim,
                                          activation=self.leaky_relu, scope='reduced_text_embeddings')

        # Concat reduced_text_embedding to conv4.
        conv4_concat = self.concat_conv_vec_and_cond_vec(conv_vec=conv4, cond_vec=reduced_text_embeddings)

        # NOTE: No pooling. Stride 1.
        # BUG: conv4_filter_dim + reduced_text_embeddings_dim??????????
        conv5 = self.conv(inputs=conv4_concat, filter_shape=[1, 1, conv4_filter_dim + self.reduced_text_embedding_dim, conv5_filter_dim],
                          pool=False, activation=self.leaky_relu_batch_norm, stride=[1, 1, 1, 1], scope='conv5')

        conv5_flat = flatten(conv5)

        fc6_logits = self.fc(inputs=conv5_flat, output_dim=1, activation=self.linear, scope='fc6')
        fc6 = tf.nn.sigmoid(fc6_logits)

        return fc6, fc6_logits

    def __build_model(self):
        """."""
        logger.info("Entry")
        self.g = tf.Graph()
        with self.g.as_default():
            self.real_images = tf.placeholder(tf.float32, name='real_images',
                                              shape=[None, self.height, self.width, self.n_channels])
            self.wrong_images = tf.placeholder(tf.float32, name='wrong_images',
                                               shape=[None, self.height, self.width, self.n_channels])
            self.real_captions = tf.placeholder(tf.float32, name='wrong_images',
                                                shape=[None, self.captions_dim])
            self.z = tf.placeholder(tf.float32, name='z',
                                    shape=[None, self.z_dim])
            # self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            with tf.variable_scope("G"):
                self.gen_images = self.__build_generator(z=self.z, text_embedding=self.real_captions)

            # NOTE: Not clipping.
            with tf.variable_scope("D") as scope:
                self.disc_real, self.disc_real_logits = self.__build_discriminator(images=self.real_images,
                                                                                   text_embeddings=self.real_captions)
                scope.reuse_variables()
                self.disc_wrong, self.disc_wrong_logits = self.__build_discriminator(images=self.wrong_images,
                                                                                     text_embeddings=self.real_captions)
                scope.reuse_variables()
                self.disc_gen, self.disc_gen_logits = self.__build_discriminator(images=self.gen_images,
                                                                                 text_embeddings=self.real_captions)

            # Losses
            # Generator Loss
            self.loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_gen_logits,
                                                                                 labels=tf.zeros_like(self.disc_gen)))

            # Discriminator Loss
            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_real_logits,
                                                                                 labels=tf.ones_like(self.disc_real)))
            d_loss_wrong = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_wrong_logits,
                                                                                  labels=tf.zeros_like(self.disc_wrong)))
            d_loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_gen_logits,
                                                                                labels=tf.zeros_like(self.disc_gen)))
            self.loss_D = d_loss_real + d_loss_wrong + d_loss_gen

            # Trainable variables
            self.vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="G")
            self.vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="D")

            # Optimizers
            optimizer_G = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
            optimizer_D = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)

            # Optimize
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
        batch_z = np.random.normal(size=[batch_size, self.z_dim])
        return batch_z

    def get_fake_batches(self, batch_size, keep_prob=1.0):
        """."""
        batch_fx, = self.sess.run([self.G], feed_dict={self.y: self.plot_batch_y,
                                                       self.z: self.plot_batch_z})
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

    def plot_fake_data(self, epoch, batch_fx, fig_plots=(4, 4), figsize=(16, 16)):
        """."""
        batch_fx = batch_fx.reshape(-1, 128, 128, 3)
        plt.figure(figsize=figsize)
        for i in range(batch_fx.shape[0]):
            plt.subplot(fig_plots[0], fig_plots[1], i + 1)
            plt.imshow(batch_fx[i], interpolation='nearest')
            plt.axis('off')
        plt.tight_layout()
        img_dir = get_project_directory("mscoco", "results", self.start_timestamp)
        plt.savefig(os.path.join(img_dir, "mscoco_gan_cnn_{}.png".format(epoch)))
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

    def run(self, epochs, summary_epochs=1):
        """."""
        batch_size = self.batch_size
        self.mscoco = mscoco_generator(subset='val', batch_size=batch_size, batch_type='captions', image_size=128)
        self.start_timestamp = get_current_timestamp()
        start_time = time.time()

        for i in range(epochs):
            logger.debug("EPOCH: {}".format(i))
            # Train discriminator
            real_images, real_captions = self.mscoco.next_batch(real=True)
            wrong_images, wrong_captions = self.mscoco.next_batch(real=False)

            batch_z = self.get_noise_batches(batch_size)
            self.sess.run([self.opt_D], feed_dict={self.real_images: real_images, self.real_captions: real_captions,
                                                   self.wrong_images: wrong_images, self.z: batch_z})

            # Train generator
            # real_images, real_captions = self.mscoco.next_batch(real=True)
            # batch_z = self.get_noise_batches(batch_size)
            self.sess.run([self.opt_G], feed_dict={self.real_images: real_images,
                                                   self.real_captions: real_captions,
                                                   self.z: batch_z})

            self.sess.run([self.opt_G], feed_dict={self.real_images: real_images,
                                                   self.real_captions: real_captions,
                                                   self.z: batch_z})

            if i % summary_epochs == 0:
                if not self.std_test_data:
                    std_real_images, std_real_captions = self.mscoco.next_batch(real=True)
                    std_wrong_images, std_wrong_captions = self.mscoco.next_batch(real=False)
                    std_batch_z = self.get_noise_batches(batch_size=self.batch_size)
                    self.std_test_data = True

                loss_G, loss_D, batch_fx = self.sess.run([self.loss_G, self.loss_D, self.gen_images],
                                                         feed_dict={self.real_images: std_real_images,
                                                                    self.real_captions: std_real_captions,
                                                                    self.wrong_images: std_wrong_images,
                                                                    self.z: std_batch_z})
                self.plot_losses(epoch=i, loss_G=loss_G, loss_D=loss_D)
                self.plot_fake_data(epoch=i, batch_fx=batch_fx)
                time_diff = time.time() - start_time
                start_time = time.time()
                logger.info("Epoch: {:3d} - L_G: {:0.3f} - L_D: {:0.3f} - Time: {:0.1f}"
                            .format(i, loss_G, loss_D, time_diff))


if __name__ == "__main__":
    setup_logger(log_directory=get_project_directory("mscoco", "logs"),
                 file_handler_type=HandlerType.TIME_ROTATING_FILE_HANDLER,
                 allow_console_logging=True,
                 allow_file_logging=True,
                 max_file_size_bytes=10000,
                 change_log_level=None)
    mnist_gan_mlp = GAN_CNN(z_dim=10, batch_size=16)
    mnist_gan_mlp.run(epochs=10000, summary_epochs=5)
