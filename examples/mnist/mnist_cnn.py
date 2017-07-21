"""."""

import os

import tensorflow as tf
from no_imagination.baseops import BaseOps
from no_imagination.utils import get_project_directory
from tensorflow.contrib.layers import flatten
from tensorflow.examples.tutorials.mnist import input_data


class CNN(BaseOps):
    """."""

    def __init__(self):
        """."""
        mnist_data_dir = get_project_directory("mnist", "dataset")
        self.mnist_data = input_data.read_data_sets(mnist_data_dir, one_hot=True)
        self.__build_model()
        self.__build_accuracy_computation()
        self.__start_session()

    def __build_model(self):
        """."""
        self.g = tf.Graph()
        with self.g.as_default():
            with tf.variable_scope("inputs"):
                self.x = tf.placeholder(tf.float32, shape=[None, 784])
                self.y = tf.placeholder(tf.float32, shape=[None, 10])
                self.keep_prob = tf.placeholder(tf.float32)
                x_image = tf.reshape(self.x, [-1, 28, 28, 1])

            with tf.variable_scope("conv1"):
                conv1 = self.conv(inputs=x_image, filter_shape=[5, 5, 1, 32], activation=tf.nn.relu, stride=[1, 1, 1, 1],
                                  pool=True, pool_stride=[1, 2, 2, 1])

            with tf.variable_scope("conv2"):
                conv2 = self.conv(inputs=conv1, filter_shape=[5, 5, 32, 64], activation=tf.nn.relu, stride=[1, 1, 1, 1],
                                  pool=True, pool_stride=[1, 2, 2, 1])

            with tf.variable_scope("fc1"):
                conv2_flat = flatten(conv2)
                fc1 = self.fc(inputs=conv2_flat, output_dim=1024, activation=tf.nn.relu, keep_prob=self.keep_prob)

            with tf.variable_scope("fc2"):
                self.y_logit = self.fc(inputs=fc1, output_dim=10, activation="linear", keep_prob=self.keep_prob)

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

    def run(self, epochs=20000, batch_size=50, keep_prob=0.5, summary_epochs=500):
        """."""
        for i in range(epochs):
            batch_x, batch_y = self.mnist_data.train.next_batch(batch_size)
            self.sess.run(self.train_step, feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: keep_prob})
            if i % summary_epochs == 0:
                print(self.sess.run(self.accuracy, feed_dict={self.x: self.mnist_data.test.images,
                                                              self.y: self.mnist_data.test.labels,
                                                              self.keep_prob: 1.0}))


if __name__ == "__main__":
    mnist_cnn = CNN()
    mnist_cnn.run(epochs=20000, batch_size=50, summary_epochs=500)
