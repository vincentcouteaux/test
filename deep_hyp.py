import numpy as np
import tensorflow as tf
from parserythm import *
from mape import *
""" the longest hypnogram of the train_input.csv file is 1311
longest of test_input is 1217"""

train_hyp, train_eegs, train_devices, train_labels, eval_hyp, eval_eegs, eval_devices, eval_labels = train_eval_base()

def pad_to1311(hyp):
    topad = 1311 - hyp.size
    out = []
    for i in range(topad):
        out.append(np.pad(hyp, (i, 1311-hyp.size-i), 'constant'))
    return out

def pad_all(hyps, labels):
    out = []
    ages = []
    for i, hyp in enumerate(hyps):
        padded = pad_to1311(hyp)
        out += padded
        for k in range(len(padded)):
            ages.append(labels[i])
    return np.array(out), np.array(ages)

def pad_right(hyps):
    out = []
    for hyp in hyps:
        out.append(np.pad(hyp,(0, 1311-hyp.size), 'constant'))
    return np.array(out)

t_hyps_pad, t_ages = pad_all(train_hyp, train_labels)
print(t_hyps_pad.shape)
e_hyps_pad = pad_right(eval_hyp)

def weight_variable(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

def bias_variable(shape, name):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool(x, n, m):
    return tf.nn.max_pool(x, ksize=[1, n, m, 1], strides=[1, n, m, 1], padding='SAME')

hyp_ph = tf.placeholder(tf.float32, shape=[None, 1311])
ages_ph = tf.placeholder(tf.float32, shape=[None])

hyp_rs = tf.reshape(hyp_ph, [-1, 1311, 1, 1])

W1 = weight_variable([50, 1, 1, 20], name="W1")
b1 = bias_variable([20], name="b1")

h_conv1 = tf.nn.relu(conv2d(hyp_rs, W1) + b1)
h_pool1 = max_pool(h_conv1, 4, 1)

W2 = weight_variable([50, 1, 20, 100], name="W2")
b2 = bias_variable([100], name="b2")

h_conv2 = tf.nn.relu(conv2d(h_pool1, W2) + b2)
h_pool2 = max_pool(h_conv2, 4, 1)

h_flat2 = tf.reshape(h_pool2, [-1, 67*100])

W3 = weight_variable([67*100, 500], "W3")
b3 = bias_variable([500], "b3")

h_fc1 = tf.nn.relu(tf.matmul(h_flat2, W3) + b3)

W4 = weight_variable([500, 1], "W4")
b4 = bias_variable([1], "b4")

out_tensor = tf.nn.relu(tf.matmul(h_fc1, W4) + b4)

euc_distance = tf.reduce_mean(tf.square(out_tensor - ages_ph))
train_step = tf.train.AdamOptimizer(1e-4).minimize(euc_distance)

def forward_batch(n, sess):
    total = t_hyps_pad.shape[0]
    r = np.random.permutation(total)
    hyps = t_hyps_pad[r]
    ages = t_ages[r]
    feed = {hyp_ph:hyps[:n], ages_ph:ages[:n]}
    dist, _ = sess.run((euc_distance, train_step), feed_dict=feed)
    #print(dist)

def eval1patient(sess, hyp):
    hyps = pad_to1311(hyp)
    ans = sess.run(out_tensor, {hyp_ph:hyps})
    return np.mean(ans), np.std(ans)
    #print("CNN; score: {}".format(mape(ans, ages)))

def eval_all(sess, hyps, ages):
    ages_c = np.zeros(ages.size)
    stds = np.zeros(ages.size)
    for i, s in enumerate(hyps):
        ages_c[i], stds[i] = eval1patient(sess, s)
    print("CNN; score: {}, std: {}".format(mape(ages_c, ages), np.mean(stds)))

def eval(sess):
    ages = sess.run(out_tensor, {hyp_ph:e_hyps_pad})
    print('CNN eeg: {}'.format(mape(ages,eval_labels)))


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for k in range(10000):
    forward_batch(50, sess)
    if k % 100 == 0:
        #eval_all(sess, eval_hyp, eval_labels)
        eval(sess)



