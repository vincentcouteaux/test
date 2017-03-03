import numpy as np
import os
from cvxopt import solvers, matrix

""" A SVM QP dual is :
maximise over alpha 2alpha.T y - alpha.T K alpha
subject to 0 <= y_i alpha_i <= 1/(2lambda.n) """

class Svm:
    def __init__(this, lambda_, kernel):
        this._lambda = lambda_
        this._kernel = kernel

    def _svm_to_qp(this, y):
        P = this.K
        n = this.K.shape[0]
        q = -y
        G = np.concatenate((np.diag(y), -np.diag(y)))
        h = np.concatenate((np.ones(y.size)/2/this._lambda/n, np.zeros(y.size)))
        return matrix(P), matrix(q), matrix(G), matrix(h)

    def _train_svm(this, y):
        Q, p, G, h = this._svm_to_qp(y)
        sol = solvers.qp(Q, p, G, h)['x']
        return np.array(sol)

    @staticmethod
    def _gram(X, dist):
        K = np.zeros((X.shape[0], X.shape[0]))
        for i, x in enumerate(X):
            for j, y in enumerate(X):
                K[i, j] = dist(x, y)
        return K

    def fit(this, X, y):
        this.K = this._gram(X, this._kernel)
        this.X = X
        this._alpha = this._train_svm(y)

    def predict(this, X):
        out = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            for j, xi in enumerate(this.X):
               out[i] += this._alpha[j]*this._kernel(x, xi)
        return np.sign(out)

    def predict_smooth(this, X):
        out = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            for j, xi in enumerate(this.X):
               out[i] += this._alpha[j]*this._kernel(x, xi)
        return out

class Multiclass_svm:
    def __init__(this, lambda_, kernel):
        this._lambda = lambda_
        this._kernel = kernel

    def fit(this, X, y):
        """ y is a matrix of shape (number_of_sample, number of classes)
        with 1 and -1 """
        this.n_classes = y.shape[1]
        this.svms = [Svm(this._lambda, this._kernel)]
        this.svms[0].fit(X, y[:, 0])
        this.K = this.svms[0].K
        this.X = X
        for i in range(1, this.n_classes):
            this.svms.append(Svm(this._lambda, this._kernel))
            this.svms[-1].K = this.K
            this.svms[-1].X = this.X
            this.svms[-1]._alpha = this.svms[-1]._train_svm(y[:, i])

    def _dist_test_train(this, Xte, X):
        out = np.zeros((Xte.shape[0], X.shape[0]))
        for i, x in enumerate(Xte):
            for j, y in enumerate(X):
                out[i, j] = this._kernel(x, y)
        return out

    #def predict(this, X):
    #    out = np.zeros((X.shape[0], this.n_classes))
    #    for i, svm in enumerate(this.svms):
    #        out[:, i] = svm.predict_smooth(X)
    #    print(out)
    #    return np.argmax(out, 1)
    def predict(this, X):
        n_classes = len(this.svms)
        Alpha = np.zeros((this.X.shape[0], n_classes))
        for i, svm in enumerate(this.svms):
            Alpha[:, i] = svm._alpha[:, 0]
        print("computing L matrix... pid={}".format(os.getpid()))
        L = this._dist_test_train(X, this.X)
        print("computing output matrix... pid={}".format(os.getpid()))
        return np.argmax(np.dot(L, Alpha), 1)


def exp_euc(x, y, sigma2):
    return np.exp(-np.sum((x-y)**2)/sigma2)

if __name__ == "__main__":
    mu1 = (2, 3, 4)
    mu2 = (1, 2, -1)
    sigma1 = 4.
    sigma2 = 4.
    X = sigma1*np.random.randn(250, 3) + mu1
    y = np.ones(250)
    X = np.concatenate((X, sigma1*np.random.randn(250, 3) + mu2))
    y = np.concatenate((y, -np.ones(250)))
    r = np.random.permutation(500)
    X = X[r]
    y = y[r]
    y = np.stack((y, -y)).T
    Xtrain, ytrain, Xtest, ytest = X[:400], y[:400], X[400:], y[400:]
    classif = Multiclass_svm(1., lambda x, y: exp_euc(x, y, 16.))
    classif.fit(Xtrain, ytrain)
    fc = classif.predict(Xtest)
    print(fc)
    print(np.argmax(ytest, 1))
    print("accuracy : {}".format(np.mean(fc == np.argmax(ytest, 1))))


