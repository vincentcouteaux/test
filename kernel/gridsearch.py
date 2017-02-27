import numpy as np
import matplotlib.pyplot as plt
from parse import *
from svm import *
from wavelet import *

def to_grey(im):
    out = np.sum(im, 3)/3.
    return out

def labels2mat(labs):
    out = -np.ones((labs.size, int(np.max(np.unique(labs))+1)))
    print(out.shape)
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


wt, sc = daubechies(2)
#images = traindb()
#images = to_grey(images)
#images = scat_and_concat(images, wt, sc, 3, lambda x: x * (x > 0))
images = csvread("train_scat_morlet_m2.csv")
train_size = 4000
t_im = images[:train_size]
e_im = images[train_size:]
labels = retrieve_labels()
t_lab = labels[:train_size]
e_lab = labels[train_size:]

bestScore = 0.
for C in [0.0001, 0.001, 0.01]:
    for sigma in [500., 1000., 5000., 10000., 15000., 50000.]:
        regr = Multiclass_svm(C, lambda x, y : exp_euc(x, y, sigma))
        regr.fit(t_im, labels2mat(t_lab))
        y_ = regr.predict(e_im)
        score = np.mean(y_ == e_lab)
        print("C={}, sigma={}, score: {}".format(C, sigma, np.mean(y_ == e_lab)))
        if score > bestScore:
            bestC = C
            bestSigma = sigma
            bestScore = score

print("BEST !!! C={}, sigma={}, score: {}".format(bestC, bestSigma, bestScore))

