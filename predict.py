from parserythm import *
from mape import *
from linear import *
from eeg_reg import *
from cycle import *
from mnetest import *
import numpy as np
import sklearn as sk
import sklearn.linear_model
import sklearn.ensemble

train_hyp, train_eegs, train_devices, train_labels, eval_hyp, eval_eegs, eval_devices, eval_labels = train_eval_base()

def extract_features(hyp, devices, eegs):
    hyp = list_max_gliss(hyp, 2)
    features = get_features(hyp)
    pools = max_pools(eegs, 30, 5)
    print(features.shape)
    print(pools.shape)
    return np.c_[(features, pools)]
    #return np.c_[(features, devices, maxf(eegs), spectrums(eegs, 30., 10., 512)/(maxf(eegs)[:, None]))]

train_features = extract_features(train_hyp, train_devices, train_eegs)
eval_features = extract_features(eval_hyp, eval_devices, eval_eegs)

regr = sk.ensemble.RandomForestRegressor()
regr.fit(train_features, train_labels)
print("score: {}".format(mape(regr.predict(eval_features), eval_labels)))

test_hyp, test_eegs = test_base()
test_features = extract_features(test_hyp, 1, test_eegs)
test_to_csv(regr.predict(test_features), 'ans3')

print("score: {}".format(mape(regr.predict(eval_features), eval_labels)))
