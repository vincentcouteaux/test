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

spec = tf.placeholder(tf.float32, shape=[None, 40, 60]) #40 frequency bins, 334 temp bins
#spec = tf.placeholder(tf.float32, shape=[None, 40, 334]) #40 frequency bins, 334 temp bins
hyp_features = tf.placeholder(tf.float32, shape=[None, 5]) # 5 is the number of hypnograms features
true_ages = tf.placeholder(tf.float32, shape=[None])

W1 = weight_variable([1, 13, 1, 5], name="W1")
b1 = bias_variable([5], name="b1")

r_spec = tf.reshape(spec, [-1, 40, 60, 1], name="reshape1")
h_conv1 = tf.nn.relu(conv2d(r_spec, W1) + b1)
h_pool1 = max_pool(h_conv1, 1, 2)

W2 = weight_variable([40, 9, 5, 5], name="W2")
b2 = bias_variable([5], name="b2")

h_conv2 = tf.nn.relu(conv2d(h_pool1, W2) + b2)
h_pool2 = max_pool(h_conv2, 1, 2)

W3 = weight_variable([1, 8, 5, 5], name="W3")
b3 = bias_variable([5], name="b3")

h_conv3 = tf.nn.relu(conv2d(h_pool2, W3) + b3)
h_pool3 = max_pool(h_conv3, 1, 2)

h_flat3 = tf.reshape(h_pool3, [-1, 5], name="reshape2")
#h_flat3 = tf.reshape(h_pool3, [-1, 35*5], name="reshape2")
h_concat = tf.concat(1, [h_flat3, hyp_features])

W4 = weight_variable([5+5, 512], "W4")
#W4 = weight_variable([35*5+5, 512], "W4")
b4 = bias_variable([512], "b4")

h_fc1 = tf.nn.relu(tf.matmul(h_concat, W4) + b4)

W5 = weight_variable([512, 1], "W5")
b5 = bias_variable([1], "b5")
ages_tensor = tf.nn.relu(tf.matmul(h_fc1, W5) + b5)

euc_distance = tf.reduce_mean(tf.square(ages_tensor - true_ages))
train_step = tf.train.AdamOptimizer(1e-4).minimize(euc_distance)

first_patient = 0
def forward_batch(n, sess, eegs, hyps, ages):
    global first_patient
    total = eegs.shape[0]
    feed = {spec: eegs[first_patient:first_patient+n], true_ages:ages[first_patient:first_patient+n], hyp_features:hyps[first_patient:first_patient+n]}
    first_patient = (first_patient+n)%total
    agest, dist, _ = sess.run((ages_tensor, euc_distance, train_step), feed_dict=feed)
    print(dist, agest)

def eval1patient(sess, spec_, hyp):
    slices = np.array(slice_specgram(spec_, 60, 15))
    hypf = []
    for k in range(len(slices)):
        hypf.append(hyp)
    hypf = np.array(hypf)
    ans = sess.run(ages_tensor, {spec:slices, hyp_features:hypf})
    return np.mean(ans), np.std(ans)
    #print("CNN; score: {}".format(mape(ans, ages)))

def eval_all(sess, specs, hyps, ages):
    ages_c = np.zeros(ages.size)
    stds = np.zeros(ages.size)
    for i, s in enumerate(specs):
        ages_c[i], stds[i] = eval1patient(sess, s, hyps[i])
    print("CNN; score: {}, std: {}".format(mape(ages_c, ages), np.mean(stds)))

def slice_specgram(specgram, size, hop):
    out = []
    for i in range((specgram.shape[1] - size)/hop + 1):
        out.append(specgram[:, i*hop:i*hop+size])
    return out

def slice_and_stack(specgrams, hyp_feat, size, hop):
    out = []
    hypf = []
    for i, s in enumerate(specgrams):
        slices = slice_specgram(s, size, hop)
        out += slices
        for k in range(len(slices)):
            hypf.append(hyp_feat[i])
    return np.array(out), np.array(hypf)

if __name__ == "__main__":
    train_hyp, train_eegs, train_devices, train_labels, eval_hyp, eval_eegs, eval_devices, eval_labels = train_eval_base()
    specs = stfts(train_eegs, 40.)
    train_feat = get_features(train_hyp)
    t_slices, t_feat = slice_and_stack(specs, train_feat, 60, 15)
    eval_specs = stfts(eval_eegs, 40.)
    eval_feat = get_features(eval_hyp)
    ages = train_labels
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(first_patient)
    for k in range(10000):
        #forward_batch(50, sess, specs, train_feat, train_labels)
        forward_batch(50, sess, t_slices, t_feat, train_labels)
        if k % 100 == 0:
            #forward_eval(sess, eval_specs, eval_feat, eval_labels)
            eval_all(sess, eval_specs, eval_feat, eval_labels)

