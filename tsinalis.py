import numpy as np
import tensorflow as tf
from eeg_reg import *
from parserythm import *
from mape import *
from linear import *

def weight_variable(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

def bias_variable(shape, name):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool(x, n, m):
    return tf.nn.max_pool(x, ksize=[1, n, m, 1], strides=[1, n, m, 1], padding='SAME')

eeg = tf.placeholder(tf.float32, shape=[None, 15000])
true_ages = tf.placeholder(tf.float32, shape=[None])

W1 = weight_variable([200, 1, 1, 20], name="W1")
b1 = bias_variable([20], name="b1")

r_eeg = tf.reshape(eeg, [-1, 15000, 1, 1])

h_conv1 = tf.nn.relu(conv2d(r_eeg, W1) + b1)
h_pool1 = max_pool(h_conv1, 20, 1)

W2 = weight_variable([30, 1, 20, 400], name="W2")
b2 = bias_variable([400], name="b2")

h_conv2 = tf.nn.relu(conv2d(h_pool1, W2) + b2)
h_pool2 = max_pool(h_conv2, 10, 1)

h_flat2 = tf.reshape(h_pool2, [-1, 72*400])

W3 = weight_variable([72*400, 500], "W3")
b3 = bias_variable([500], "b3")

h_fc1 = tf.nn.relu(tf.matmul(h_flat2, W3) + b3)

W4 = weight_variable([500, 500], "W4")
b4 = bias_variable([500], "b4")

h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W4) + b4)

W5 = weight_variable([500, 1], "W5")
b5 = bias_variable([1], "b5")
ages_tensor = tf.nn.relu(tf.matmul(h_fc1, W5) + b5)

euc_distance = tf.reduce_mean(tf.square(ages_tensor - true_ages))
train_step = tf.train.AdamOptimizer(1e-4).minimize(euc_distance)


def slice_eeg(eeg, size, hop):
    out = []
    for i in range((eeg.size - size)/hop + 1):
        out.append(eeg[i*hop:i*hop+size])
    return out



a = np.ones((50,15000))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(train_step, {eeg:a, true_ages:np.ones(50)})
