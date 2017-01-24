import numpy as np
from parserythm import *
import sklearn as sk
import sklearn.ensemble
import pywt
from mape import *

def get_portions(hyp, n_points, hop):
    out = [];
    for k in range((hyp.size - n_points)/hop + 1):
        portion = hyp[k*hop:(k*hop + n_points)]
        coefs = pywt.wavedec(portion, 'db1')
        out.append(np.concatenate(coefs))
    return out

def all_portions(hyps, n_points, hop, labels):
    out = [];
    ages = [];
    for i, hyp in enumerate(hyps):
        portions = get_portions(hyp, n_points, hop)
        out += portions
        age = labels[i]
        for k in portions:
            ages.append(age)
    return out, ages

def predict(hyp, n_points, hop, regr):
    batch = np.array(get_portions(hyp, n_points, hop))
    if hyp.size < 200:
        return []
    ages = regr.predict(batch)
    return np.mean(ages)

def predict_all(hyps, n_points, hop, regr):
    out = []
    for hyp in hyps:
        out.append(predict(hyp, n_points, hop, regr))
    for i, v in enumerate(out):
        if v == []:
            out[i] = 40
    return np.array(out)

if __name__ == "__main__":
    train_hyp, train_eegs, train_devices, train_labels, eval_hyp, eval_eegs, eval_devices, eval_labels = train_eval_base()
    batch, ages = all_portions(train_hyp, 200, 50, train_labels)
    regr = sk.ensemble.RandomForestRegressor()
    regr.fit(np.array(batch), ages)
    print("score: {}".format(mape(predict_all(eval_hyp, 200, 50, regr), eval_labels)))




