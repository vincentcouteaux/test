import numpy as np
import matplotlib.pyplot as plt
from wavelet import *
from tensorflow.examples.tutorials.mnist import input_data
from svm import *

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
wt, sc = daubechies(2)

X, labels = mnist.train.next_batch(700)
#labels = 2*labels[:, 7] - 1
labels = 2*labels - 1
Xtrain, ytrain, Xtest, ytest = X[:500], labels[:500], X[500:], labels[500:]
relu = lambda x: x * (x > 0)
Xtrain_scat = scat_and_concat(np.reshape(Xtrain, (-1, 28, 28)), wt, sc, 2, relu)
Xtest_scat = scat_and_concat(np.reshape(Xtest, (-1, 28, 28)), wt, sc, 2, relu)

classif = Multiclass_svm(.001, lambda x, y: exp_euc(x, y, 10.))
classif.fit(Xtrain_scat, ytrain)
y = classif.predict(Xtest_scat)
#classif.fit(Xtrain, ytrain)
#y = classif.predict(Xtest)
print(y)
print("accuracy : {}".format(np.mean(y == np.argmax(ytest, 1))))
miss = np.where(y == np.argmax(ytest, 1))[0]
for m in miss:
    plt.figure()
    plt.imshow(np.reshape(Xtest[m], (28, 28)), interpolation="none")
    plt.figure()
    plt.imshow(np.reshape(Xtrain_scat[m], (28, 28)), interpolation="none")
    print("true: {}, forecasted: {}".format(np.argmax(ytest[m]), y[m]))
    print(ytest[m])
    plt.show()

