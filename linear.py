import numpy as np
from parserythm import *
import pywt

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

def wavelet(eeg):
    coefs, _ = pywt.cwt(eeg, np.arange(1, 129), 'gaus1')
    plt.imshow(coefs, aspect="auto")
    plt.show()

if __name__ == "__main__":
    hypnograms = get_hypnograms('train_input.csv')
    labels = get_labels('challenge_output_data_training_file_age_prediction_from_eeg_signals.csv')
    #eegs = get_eegs('train_input.csv')
    #wavelet(eegs[0])
    deepsleep = get_deep_sleep_proportion(hypnograms)
    awake = get_awake_proportion(hypnograms)
    rem = get_rem_time(hypnograms)
    plt.scatter(deepsleep, labels)
    plt.title('deep sleep proportion')
    plt.figure()
    plt.scatter(rem, labels)
    plt.title('rem time')
    plt.show()
