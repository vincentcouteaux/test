import numpy as np
from parserythm import *

def get_deep_sleep_proportion(hypnogram):
    return float(np.sum(hypnogram == 3))/hypnogram.size

if __name__ == "__main__":
    hypnograms = get_hypnograms('train_input.csv')
    labels = get_labels('challenge_output_data_training_file_age_prediction_from_eeg_signals.csv')
    averages = np.zeros(len(hypnograms))
    for i, k in enumerate(hypnograms):
        averages[i] = get_deep_sleep_proportion(k)
    plt.scatter(averages, labels)
    plt.show()

