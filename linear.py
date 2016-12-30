import numpy as np
from parserythm import *
import pywt

#test

def get_deep_sleep_proportion(hypnogram):
    return float(np.sum(hypnogram == 3))/hypnogram.size

def wavelet(eeg):
    coefs, _ = pywt.cwt(eeg, np.arange(1, 129), 'gaus1')
    plt.matshow(coefs)
    plt.show()

if __name__ == "__main__":
    hypnograms = get_hypnograms('train_input.csv')
    labels = get_labels('challenge_output_data_training_file_age_prediction_from_eeg_signals.csv')
    eegs = get_eegs('train_input.csv')
    wavelet(eegs[0])
