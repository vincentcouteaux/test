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

first_patient = 0

def forward_batch(n, sess, eegs, ages):
    global first_patient
    total = eegs.shape[0]
    feed = {eegs_: eegs[first_patient:first_patient+n], true_ages:ages[first_patient:first_patient+n]}
    first_patient = (first_patient+n)%total
    agest, dist, _ = sess.run((ages_tensor, euc_distance, train_step), feed_dict=feed)
    #print(dist, agest)

def slice_eeg(eeg, size, hop):
    out = []
    for i in range((eeg.size - size)/hop + 1):
        out.append(eeg[i*hop:i*hop+size])
    return out

def eval1patient(sess, eeg):
    slices = np.array(slice_eeg(eeg, 15000, 5000))
    ans = sess.run(ages_tensor, {eeg:slices})
    return np.mean(ans), np.std(ans)
    #print("CNN; score: {}".format(mape(ans, ages)))

def eval_all(sess, eegs, ages):
    ages_c = np.zeros(ages.size)
    stds = np.zeros(ages.size)
    for i, s in enumerate(eegs):
        ages_c[i], stds[i] = eval1patient(sess, s)
    print("CNN; score: {}, std: {}".format(mape(ages_c, ages), np.mean(stds)))

def slice_and_stack(eegs, size, hop):
	out = []
	for s in eegs:
	    slices = slice_eeg(s, size, hop)
	    out += slices
	return np.array(out)

if __name__ == "__main__":
    train_hyp, train_eegs, train_devices, train_labels, eval_hyp, eval_eegs, eval_devices, eval_labels = train_eval_base()
    t_slices = slice_and_stack(train_eegs, 15000, 5000)
    ages = train_labels
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(first_patient)
    for k in range(10000):
        #forward_batch(50, sess, specs, train_feat, train_labels)
        forward_batch(50, sess, t_slices, train_labels)
        if k % 100 == 0:
            #forward_eval(sess, eval_specs, eval_feat, eval_labels)
            eval_all(sess, eval_eegs, eval_labels)