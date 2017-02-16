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
import sklearn.kernel_ridge
import sklearn.svm

train_hyp, train_eegs, train_devices, train_labels, eval_hyp, eval_eegs, eval_devices, eval_labels = train_eval_base()
#test_hyp, test_eegs = test_base()
#train_eegs = get_eegs('train_input.csv')
#train_hyp = get_hypnograms('train_input.csv')
#train_labels = get_labels('challenge_output_data_training_file_age_prediction_from_eeg_signals.csv')

def extract_features(hyp, devices, eegs):
    #hyp = list_max_gliss(hyp, 2)
    features = get_features(hyp)
    pools = max_pools(eegs, 100, 10)
    #pools = norms_psds(eegs, 40)
    return np.c_[(features, pools)]
    #return np.c_[(features, devices, maxf(eegs), spectrums(eegs, 30., 10., 512)/(maxf(eegs)[:, None]))]

train_features = extract_features(train_hyp, 1, train_eegs)
eval_features = extract_features(eval_hyp, eval_devices, eval_eegs)
#test_features = extract_features(test_hyp, 1, test_eegs)

#regr = sk.kernel_ridge.KernelRidge(kernel="rbf", gamma=0.000001, alpha=0.00001)
for k in range(1, train_features.shape[1]):
    regr = sk.ensemble.RandomForestRegressor(max_features=k, n_estimators=100)
    if not np.isfinite(np.sum(train_features)):
        print("INFINITE !!!!")
    regr.fit(train_features, train_labels)
    print("*** max_features = {}, train score: {}, eval score: {}".format(k, mape(regr.predict(train_features), train_labels), mape(regr.predict(eval_features), eval_labels)))
print("\n\n %%%%%%%%%%% NUMBER OF TREES %%%%%%%%%%%% \n\n")
for k in range(100, 150):
    regr = sk.ensemble.RandomForestRegressor(n_estimators=k)
    regr.fit(train_features, train_labels)
    print("*** n_estimators = {}, train score: {}, eval score: {}".format(k, mape(regr.predict(train_features), train_labels), mape(regr.predict(eval_features), eval_labels)))

print("\n\n %%%%%%%%%%% MAX depth %%%%%%%%%%%% \n\n")

for k in range(3, 50):
    regr = sk.ensemble.RandomForestRegressor(n_estimators=100, max_depth=k)
    regr.fit(train_features, train_labels)
    print("*** max_depth = {}, train score: {}, eval score: {}".format(k, mape(regr.predict(train_features), train_labels), mape(regr.predict(eval_features), eval_labels)))

#test_to_csv(regr.predict(test_features), 'ans4')

#print("RF; score: {}".format(mape(regr.predict(train_features), train_labels)))
