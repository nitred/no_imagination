"""."""

import os

import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.layers import Input
from no_imagination.utils import get_project_directory
from tensorflow.examples.tutorials.mnist import input_data

VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)


class CNNPretrained(object):
    """."""

    def __init__(self):
        """."""
        mnist_data_dir = get_project_directory("mnist", "dataset")
        self.mnist_data = input_data.read_data_sets(mnist_data_dir, one_hot=True)
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
                self.x = tf.placeholder(tf.float32, shape=[None, 784])
                self.y = tf.placeholder(tf.float32, shape=[None, 10])
                self.keep_prob = tf.placeholder(tf.float32)
                x_image = tf.reshape(self.x, [-1, 28, 28, 1])

            with tf.variable_scope("vgg16"):
                vgg16_input = Input(tensor=x_image)
                vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=vgg16_input,
                              input_shape=None, pooling='max')

            with tf.variable_scope("fc1"):
                h_pool2_flat = tf.reshape(vgg16, [-1, 7 * 7 * 64])
                h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
                h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

            with tf.variable_scope("fc2"):
                W_fc2 = self.__weight_variable([1024, 10])
                b_fc2 = self.__bias_variable([10])
                self.y_out = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_out, labels=self.y))
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
            self.train_step = optimizer.minimize(cross_entropy)

    def __build_accuracy_computation(self):
        """."""
        with self.g.as_default():
            # boolean prediction
            correct_prediction = tf.equal(tf.argmax(self.y_out, 1), tf.argmax(self.y, 1))
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
    mnist_cnn = CNNPretrained()
    mnist_cnn.run(epochs=20, batch_size=50, summary_epochs=500)
