from parserythm import *
from mape import *
from linear import *
from eeg_reg import *
import numpy as np
import sklearn as sk
import sklearn.linear_model
import sklearn.ensemble

train_hyp, train_eegs, train_labels, eval_hyp, eval_eegs, eval_labels = train_eval_base()

train_features = get_features(train_hyp)
#train_features = np.c_[(train_features, f_0s(train_eegs))]
train_spec = spectrums(train_eegs, 30., 256)
print(train_features.shape, train_spec.shape)
#train_features = np.c_[(train_features, train_spec)]
train_features = train_spec

eval_spec = spectrums(eval_eegs, 30., 256)
eval_features = get_features(eval_hyp)
#eval_features = np.c_[(eval_features, eval_spec)]
eval_features = eval_spec

#regr = sk.ensemble.RandomForestRegressor()
regr = sk.linear_model.LinearRegression()
regr.fit(train_features, train_labels)
print("score: {}".format(mape(regr.predict(eval_features), eval_labels)))

regr = sk.ensemble.RandomForestRegressor(n_estimators=30)
regr.fit(train_features, train_labels)
print("score: {}".format(mape(regr.predict(eval_features), eval_labels)))

