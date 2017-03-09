import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from parse import *
from svm import *
from wavelet import *

def to_grey(im):
    out = np.sum(im, 3)/3.
    return out

def labels2mat(labs):
    out = -np.ones((labs.size, int(np.max(np.unique(labs))+1)))
    #print(out.shape)
    for i, l in enumerate(labs):
        out[i, int(l)] = 1
    return out

def csvread(filename):
    out=[]
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            out.append(np.double(row))
    return np.array(out)


#wt, sc = daubechies(2)
#images = traindb()
#images = to_grey(images)
#images = scat_and_concat(images, wt, sc, 3, lambda x: x * (x > 0))
print("#reading images...")
#images = csvread("train_m12_OPP_norm.csv")
images = csvread("train_m12_hog.csv")
#mean = np.mean(images, 0)
#std = np.std(images, 0)
#images = (images - mean)/std
train_size = 4000
t_im = images[:train_size]
e_im = images[train_size:]
labels = retrieve_labels()
t_lab = labels[:train_size]
e_lab = labels[train_size:]

def test_node(C, sigma):
    regr = Multiclass_svm(C, lambda x, y : exp_euc(x, y, sigma))
    print("Computing Gram matrix...", os.getpid())
    regr.fit(t_im, labels2mat(t_lab))
    y_ = regr.predict(e_im)
    print("########### {}".format(np.mean(y_ == e_lab)))
    return np.mean(y_ == e_lab), np.mean(regr.predict_train() == t_lab)

bestScore = 0.
f = open("resultshog.txt", "w")
#tup = ((1e-6, 110000), (1e-6, 120000), (1e-6, 135000), (1e-6, 150000), (1e-6, 175000),
#        (7e-7, 110000), (7e-7, 120000), (7e-7, 135000), (7e-7, 150000), (7e-7, 175000),
#        (1e-5, 12000), (1e-5, 10000), (5e-5, 8000), (5e-5, 1000), (1e-4, 10000))
for C in [1e-5, 1e-6, 5e-6, 3e-5]:
    for sigma in [15000., 30000., 50000., 100000.,10000., 40000.]:
#for C, sigma in tup:
        if os.fork() == 0:
            score, trainscore = test_node(C, sigma)
            print("\n FINISHED!! C={}, sigma={}, score: {} -- pid={}\n\n".format(C, sigma, score, os.getpid()))
            f.write("C={}, sigma={}, score: {} -- pid={}\n".format(C, sigma, score, os.getpid()))
            print("TRAIN SCORE {}".format(trainscore))
            if score > bestScore:
                bestC = C
                bestSigma = sigma
                bestScore = score
            sys.exit(0)
f.close()

#print("BEST !!! C={}, sigma={}, score: {}".format(bestC, bestSigma, bestScore))

