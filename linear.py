import numpy as np
from parserythm import *
import sklearn as sk
import sklearn.linear_model
import pywt
from mape import *

def get_deep_sleep_proportion(hypnograms):
    out = []
    for hypnogram in hypnograms:
        out.append(np.mean(hypnogram == 3))
    return np.array(out)

def get_awake_proportion(hypnograms):
    out = []
    for hypnogram in hypnograms:
        out.append(np.mean(hypnogram == 0))
    return np.array(out)

def get_rem_time(hypnograms):
    out = []
    for hypnogram in hypnograms:
        out.append(np.sum(hypnogram == 4))
    return np.array(out)

def number_of_wakening(hypnograms):
    out = []
    for hypnogram in hypnograms:
        out.append(0)
        for i in range(hypnogram.size - 1):
            if hypnogram[i] != 0 and hypnogram[i+1] == 0:
                out[-1] += 1
    return np.array(out)

def get_waso(hypnogram):
    asleep = np.argmax(hypnogram > 0)
    return sum(hypnogram[asleep:] == 0)

def wake_after_sleep_onset(hypnograms):
    out=[]
    for hypnogram in hypnograms:
        out.append(get_waso(hypnogram))
    return np.array(out)

def total_sleep_time(hypnograms):
    out=[]
    for hypnogram in hypnograms:
        out.append(len(hypnogram))
    return np.array(out)

def number_of_deep_sleep(hypnograms):
    out = []
    for hypnogram in hypnograms:
        out.append(0)
        for i in range(hypnogram.size - 1):
            if hypnogram[i] != 3 and hypnogram[i+1] == 3:
                out[-1] += 1
    return np.array(out)

def wavelet(eeg):
    coefs, _ = pywt.cwt(eeg, np.arange(1, 129), 'gaus1')
    plt.imshow(coefs, aspect="auto")
    plt.show()

def get_features(hypnograms):
    return np.stack((get_deep_sleep_proportion(hypnograms), wake_after_sleep_onset(hypnograms), number_of_wakening(hypnograms), number_of_deep_sleep(hypnograms), total_sleep_time(hypnograms))).T

def train_and_give_forecast(X, ages):
    regr = sk.linear_model.LinearRegression()
    regr.fit(X, ages)
    return regr.predict(X)

if __name__ == "__main__":
    hypnograms = get_hypnograms('train_input.csv')
    labels = get_labels('challenge_output_data_training_file_age_prediction_from_eeg_signals.csv')
    #eegs = get_eegs('train_input.csv')
    #wavelet(eegs[0])
    deepsleep = get_deep_sleep_proportion(hypnograms)
    awake = get_awake_proportion(hypnograms)
    rem = get_rem_time(hypnograms)
    wakenings = number_of_wakening(hypnograms)
    #plt.scatter(labels, deepsleep)
    #plt.title('deep sleep proportion')
    #plt.figure()
    #plt.scatter(labels, wakenings)
    #plt.title('wakenings time')
    #plt.figure()
    #plt.scatter(labels, wake_after_sleep_onset(hypnograms))
    #plt.title('waso (min)')
    #plt.figure()
    #plt.scatter(labels, number_of_deep_sleep(hypnograms)/deepsleep)
    #plt.title('deep sleep quality')
    #plt.show()
    print(mape(train_and_give_forecast(get_features(hypnograms), labels), labels))
