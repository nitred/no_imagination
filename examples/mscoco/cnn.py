"""."""

import tensorflow as tf
from no_imagination.baseops import BaseOps
from no_imagination.datasets.mscoco import mscoco_generator
from no_imagination.models.vgg16 import Vgg16_Pretrained_Conv
from no_imagination.utils import get_project_directory, get_project_file
from tensorflow.contrib.layers import flatten
from tensorflow.examples.tutorials.mnist import input_data


class CNNPretrained(BaseOps):
    """."""

    def __init__(self, height=224, width=224, n_channels=3, n_categories=10, batch_size=32):
        """."""
        self.height = height
        self.width = width
        self.n_channels = n_channels
        self.n_categories = n_categories
        self.batch_size = batch_size

        self.val_dataset = mscoco_generator(subset='val',
                                            n_categories=n_categories,
                                            batch_size=batch_size)
        self.train_dataset = mscoco_generator(subset='train',
                                              categories=self.val_dataset.get_category_ids(),
                                              batch_size=batch_size)

        self.__build_model()
        self.__build_accuracy_computation()
        self.__start_session()

    def __weight_variable(self, shape, initializer=tf.truncated_normal_initializer(stddev=0.1)):
        """."""
        return tf.get_variable("weights", shape=shape, dtype=tf.float32, initializer=initializer)

    def __bias_variable(self, shape, initializer=tf.constant_initializer(value=0.0)):
        """."""
        return tf.get_variable("biases", shape=shape, dtype=tf.float32, initializer=initializer)

    def __conv2d(self, x, weights):
        """."""
        return tf.nn.conv2d(input=x, filter=weights, strides=[1, 1, 1, 1], padding="SAME")

    def __max_pool(self, x, window=[2, 2]):
        """."""
        return tf.nn.max_pool(x,
                              ksize=[1, window[0], window[1], 1],
                              strides=[1, window[0], window[1], 1],
                              padding="SAME")

    def __build_model(self):
        """."""
        self.g = tf.Graph()
        with self.g.as_default():
            with tf.variable_scope("inputs"):
                self.x = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.n_channels])
                self.y = tf.placeholder(tf.float32, shape=[None, self.n_categories])
                self.keep_prob = tf.placeholder(tf.float32)

            with tf.variable_scope("conv1"):
                conv1 = self.conv(inputs=self.x, filter_shape=[5, 5, 3, 32], activation=tf.nn.relu, stride=[1, 1, 1, 1],
                                  pool=True, pool_stride=[1, 2, 2, 1])

            with tf.variable_scope("conv2"):
                conv2 = self.conv(inputs=conv1, filter_shape=[5, 5, 32, 64], activation=tf.nn.relu, stride=[1, 1, 1, 1],
                                  pool=True, pool_stride=[1, 2, 2, 1])

            with tf.variable_scope("conv3"):
                conv3 = self.conv(inputs=conv2, filter_shape=[5, 5, 64, 128], activation=tf.nn.relu, stride=[1, 1, 1, 1],
                                  pool=True, pool_stride=[1, 2, 2, 1])

            with tf.variable_scope("conv4"):
                conv4 = self.conv(inputs=conv3, filter_shape=[5, 5, 128, 256], activation=tf.nn.relu, stride=[1, 1, 1, 1],
                                  pool=True, pool_stride=[1, 2, 2, 1])

            # with tf.variable_scope("vgg16"):
            #     vgg16_npy_path = get_project_file(filename='vgg16.npy', subdirs=['vgg', 'dataset'])
            #     vgg16 = Vgg16_Pretrained_Conv(vgg16_npy_path=vgg16_npy_path, vgg_input=self.x)
            #     vgg16_pool5 = vgg16.get_pretrained_conv_layer()
            #
            # with tf.variable_scope("fc1"):
            #     vgg16_pool5_flat = flatten(vgg16_pool5)
            #     fc1 = self.fc(inputs=vgg16_pool5_flat, output_dim=1024,
            #                   activation=tf.nn.relu, keep_prob=self.keep_prob)

            with tf.variable_scope("fc1"):
                conv4_flat = flatten(conv4)
                fc1 = self.fc(inputs=conv4_flat, output_dim=2048,
                              activation=tf.nn.relu, keep_prob=self.keep_prob)

            with tf.variable_scope("fc2"):
                fc2 = self.fc(inputs=fc1, output_dim=1024,
                              activation=tf.nn.relu, keep_prob=self.keep_prob)

            with tf.variable_scope("fc3"):
                self.y_logit = self.fc(inputs=fc2, output_dim=self.n_categories,
                                       activation="linear", keep_prob=self.keep_prob)

            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_logit, labels=self.y))
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
            self.train_step = optimizer.minimize(cross_entropy)

    def __build_accuracy_computation(self):
        """."""
        with self.g.as_default():
            # boolean prediction
            correct_prediction = tf.equal(tf.argmax(self.y_logit, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def __start_session(self):
        """."""
        with self.g.as_default():
            self.sess = tf.Session(graph=self.g)
            self.sess.run(tf.global_variables_initializer())

    def run(self, epochs=10, keep_prob=0.5, summary_epochs=1):
        """."""
        for i in range(epochs):
            batch_x, batch_y = self.train_dataset.next_batch()
            self.sess.run(self.train_step, feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: keep_prob})
            if i % summary_epochs == 0:
                batch_x, batch_y = self.val_dataset.test_batch(batch_size=64)
                print(self.sess.run(self.accuracy, feed_dict={self.x: batch_x,
                                                              self.y: batch_y,
                                                              self.keep_prob: 1.0}))


if __name__ == "__main__":
    cnn = CNNPretrained(batch_size=64)
    cnn.run(epochs=10000, summary_epochs=100)