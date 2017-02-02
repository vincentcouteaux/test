import numpy as np
import tensorflow as tf
from parserythm import *
""" the longest hypnogram of the train_input.csv file is 1311
longest of test_input is 1217"""

train_hyp, train_eegs, train_devices, train_labels, eval_hyp, eval_eegs, eval_devices, eval_labels = train_eval_base()

def pad_to1311(hyp):
    return np.pad(hyp, (0, 1311-hyp.size), 'constant')
def pad_all(hyps):
    out = []
    for hyp in hyps:
        out.append(pad_to1311(hyp))
    return np.array(out)

def weight_variable(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

def bias_variable(shape, name):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool(x, n, m):
    return tf.nn.max_pool(x, ksize=[1, n, m, 1], strides=[1, n, m, 1], padding='SAME')

hyp_ph = tf.placeholder(tf.float32, shape=[None, 1311])
ages_ph = tf.placeholder(tf.float32, shape[None])

hyp_rs = tf.reshape(hyp_ph, [-1, 1311, 1, 1])

W1 = weight_variable([50, 1, 1, 20], name="W1")
b1 = bias_variable([20], name="b1")

h_conv1 = tf.nn.relu(conv2d(r_spec, W1) + b1)
h_pool1 = max_pool(h_conv1, 2, 1)

W2 = weight_variable([40, 9, 5, 5], name="W2")
b2 = bias_variable([5], name="b2")

h_conv2 = tf.nn.relu(conv2d(h_pool1, W2) + b2)
h_pool2 = max_pool(h_conv2, 1, 2)



