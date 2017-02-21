import numpy as np
import tensorflow as tf
from parserythm import *
from eeg_reg import *
from mape import *

def pad_right(hyps):
    out = []
    for hyp in hyps:
        out.append(np.pad(hyp,(0, 1311-hyp.size), 'constant'))
    return np.array(out)

def weight_variable(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

def bias_variable(shape, name):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool(x, n, m):
    return tf.nn.max_pool(x, ksize=[1, n, m, 1], strides=[1, n, m, 1], padding='SAME')

spec_ph = tf.placeholder(tf.float32, shape=[None, 40, 334]) #40 frequency bins, 334 temp bins
hyp_ph = tf.placeholder(tf.float32, shape=[None, 1311])
ages_ph = tf.placeholder(tf.float32, shape=[None])
buckets_ph = tf.placeholder(tf.float32, shape=[None, 5])
keep_prob = tf.placeholder(tf.float32)

hyp_rs = tf.reshape(hyp_ph, [-1, 1311, 1, 1])

W1_h = weight_variable([50, 1, 1, 20], name="W1")
b1_h = bias_variable([20], name="b1")

h_conv1 = tf.nn.relu(conv2d(hyp_rs, W1_h) + b1_h)
h_pool1 = max_pool(h_conv1, 4, 1)

W2_h = weight_variable([50, 1, 20, 100], name="W2")
b2_h = bias_variable([100], name="b2")

h_conv2 = tf.nn.relu(conv2d(h_pool1, W2_h) + b2_h)
h_pool2 = max_pool(h_conv2, 4, 1)

h_flat2 = tf.reshape(h_pool2, [-1, 67*100])

W1_e = weight_variable([1, 13, 1, 5], name="W1_e")
b1_e = bias_variable([5], name="b1_e")

r_spec = tf.reshape(spec_ph, [-1, 40, 334, 1], name="reshape1")
e_conv1 = tf.nn.relu(conv2d(r_spec, W1_e) + b1_e)
e_pool1 = max_pool(e_conv1, 1, 2)

W2_e = weight_variable([40, 9, 5, 5], name="W2_e")
b2_e = bias_variable([5], name="b2_e")

e_conv2 = tf.nn.relu(conv2d(e_pool1, W2_e) + b2_e)
e_pool2 = max_pool(e_conv2, 1, 2)

W3_e = weight_variable([1, 8, 5, 5], name="W3_e")
b3_e = bias_variable([5], name="b3_e")

e_conv3 = tf.nn.relu(conv2d(e_pool2, W3_e) + b3_e)
e_pool3 = max_pool(e_conv3, 1, 2)

e_flat3 = tf.reshape(e_pool3, [-1, 35*5], name="reshape2")

h_concat = tf.concat(1, [h_flat2, e_flat3])

W4 = weight_variable([67*100+35*5, 512], "W4")
b4 = bias_variable([512], "b4")

h_fc1 = tf.nn.relu(tf.matmul(h_concat, W4) + b4)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#W5 = weight_variable([512, 1], "W5")
#b5 = bias_variable([1], "w5")

buckets = 5

W5 = weight_variable([512, buckets], "W5")
b5 = bias_variable([buckets], "w5")

out_t = tf.nn.relu(tf.matmul(h_fc1_drop, W5) + b5)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=buckets_ph, logits=out_t))

#euc_distance = tf.reduce_mean(tf.square((out_t - ages_ph)))
#mape_dist = tf.reduce_mean(tf.abs((out_t - ages_ph)/ages_ph))
train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(e_pool3, {spec_ph:np.ones((5, 40, 334))}).shape)

def bucket_ages(ages, n_buckets=buckets):
    m = 18.
    M = 68.
    step = (M - m)/n_buckets
    buckets = np.floor((ages-m)/step)
    buckets[buckets > (n_buckets-1)] = (n_buckets-1) * np.ones(np.sum(buckets > (n_buckets-1)))
    buckets[buckets < 0] = np.zeros(np.sum(buckets < 0))
    print(ages.size, n_buckets)
    out = np.zeros((ages.size, n_buckets))
    for i, b in enumerate(buckets):
        out[i, int(b)] = 1
    return out

print(bucket_ages(np.arange(15, 78)))

def forward_batch(n, sess, eegs, hyps, ages):
    total = eegs.shape[0]
    r = np.random.permutation(total)
    eegs = eegs[r]
    hyps = hyps[r]
    ages = ages[r]
    ages = bucket_ages(ages)
    feed = {spec_ph: eegs[:n], buckets_ph:ages[:n], hyp_ph:hyps[:n], keep_prob:0.5}
    agest, dist, _ = sess.run((out_t, cross_entropy, train_step), feed_dict=feed)
    print(dist)
    if np.isnan(agest).any():
        print(agest)

train_hyp, train_eegs, train_devices, train_labels, eval_hyp, eval_eegs, eval_devices, eval_labels = train_eval_base()
t_hyp_padded = pad_right(train_hyp)
train_stft = stfts(train_eegs, 40.)
e_hyp_padded = pad_right(eval_hyp)
eval_stft = stfts(eval_eegs, 40.)

def get_mean_std(train_stft):
    return np.mean(train_stft, (0, 2)), np.std(train_stft, (0, 2))

def normalize(stft, mean, std):
    return (stft - np.reshape(mean, (1, -1, 1)))/np.reshape(std, (1, -1, 1))

mean, std = get_mean_std(train_stft)
train_stft = normalize(train_stft, mean, std)
eval_stft = normalize(eval_stft, mean, std)

best = 1.
eval_buckets = bucket_ages(eval_labels)
for k in range(500000):
    forward_batch(50, sess, train_stft, t_hyp_padded, train_labels)
    if k % 100 == 0:
        feed = {spec_ph: eval_stft, hyp_ph:e_hyp_padded, buckets_ph:eval_buckets, keep_prob:1}
        agest = sess.run((out_t,), feed_dict = feed)
        #print("EVAL !! : score: {}, mean: {}; min: {}, max: {}; std: {}".format(dist, np.mean(agest), np.min(agest), np.max(agest), np.std(agest)))
        dist = np.mean(np.argmax(agest) == np.argmax(eval_buckets))
        print(agest)
        print(eval_buckets)
        print(agest[0][:, 0])
        print(eval_buckets[:, 0])
        print("EVAL !! score: {}, 1st bkt={}, {}, ; 2nd bkt = {},{} ; 3rd bkt = {}, {} ; 4th bkt = {},{}".format(dist,
            np.sum(agest[0][:, 0]), np.sum(eval_buckets[:, 0]),
            np.sum(agest[0][:, 1]), np.sum(eval_buckets[:, 1]),
            np.sum(agest[0][:, 2]), np.sum(eval_buckets[:, 2]),
            np.sum(agest[0][:, 3]), np.sum(eval_buckets[:, 3]),
            np.sum(agest[0][:, 4]), np.sum(eval_buckets[:, 4])))
        if (best > dist):
            best = dist
print("BEST !!! : {}".format(best))


