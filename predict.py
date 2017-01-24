from parserythm import *
from mape import *
from linear import *
from eeg_reg import *
import numpy as np
import sklearn as sk
import sklearn.linear_model
import sklearn.ensemble

train_hyp, train_eegs, train_devices, train_labels, eval_hyp, eval_eegs, eval_devices, eval_labels = train_eval_base()

def extract_features(hyp, devices, eegs):
    features = get_features(hyp)
    return np.c_[(features, devices, maxf(eegs), spectrums(eegs, 30., 10., 512))]
    #return np.c_[(features, devices, maxf(eegs), spectrums(eegs, 30., 10., 512)/(maxf(eegs)[:, None]))]

train_features = extract_features(train_hyp, train_devices, train_eegs)
eval_features = extract_features(eval_hyp, eval_devices, eval_eegs)

regr = sk.ensemble.RandomForestRegressor()
#regr = sk.linear_model.LinearRegression()
regr.fit(train_features, train_labels)
print("score: {}".format(mape(regr.predict(eval_features), eval_labels)))

#regr = sk.ensemble.RandomForestRegressor(n_estimators=30)
#regr.fit(train_features, train_labels)
#print("score: {}".format(mape(regr.predict(eval_features), eval_labels)))

