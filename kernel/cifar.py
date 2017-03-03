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

def test_to_csv(forecast, filename):
    with open(filename + '.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['Id', 'Prediction'])
        for i in range(forecast.size):
            spamwriter.writerow([i+1, forecast[i]])

def csvread(filename):
    out=[]
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            out.append(np.double(row))
    return np.array(out)

if __name__ == "__main__":
    wt, sc = daubechies(2)
#images = traindb()
#images = to_grey(images)
#images = scat_and_concat(images, wt, sc, 4, lambda x: x * (x > 0))
    t_im = csvread("train_scat_m2_fredecOPP.csv")
    e_im = csvread("test_scat_m2_fredecOPP.csv")
    labels = retrieve_labels()
    t_lab = labels

    C = 1e-5
    sigma=30000
    regr = Multiclass_svm(C, lambda x, y : exp_euc(x, y, sigma))
    regr.fit(t_im, labels2mat(t_lab))
    y_ = regr.predict(e_im)
    test_to_csv(y_, 'Yte_m2_fredecOPP')


