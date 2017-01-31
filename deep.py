import numpy as np
import tensorflow as tf
from eeg_reg import *
from parserythm import *

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool(x, n, m):
    return tf.nn.max_pool(x, ksize=[1, n, m, 1], strides=[1, n, m, 1], padding='SAME')

spec = tf.placeholder(tf.float32, shape=[None, 40, 334]) #40 frequency bins, 334 temp bins
true_ages = tf.placeholder(th.float32, shape=[None])

W1 = weight_variable([1, 13, 1, 5])
b1 = bias_variable([5])

r_spec = tf.reshape(spec, [-1, 40, 334, 1])
h_conv1 = tf.nn.relu(conv2d(r_spec, W1) + b1)
h_pool1 = max_pool(h_conv1, 1, 2)

W2 = weight_variable([40, 9, 1, 5])
b2 = bias_variable([5])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W2) + b2)
h_pool2 = max_pool(h_conv2, 1, 2)

W3 = weight_variable([1, 8, 1, 5])
b2 = bias_variable([5])

