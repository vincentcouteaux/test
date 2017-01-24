import numpy as np
from parserythm import *

def max_gliss(hyp, size_filter):
	out = np.zeros(hyp.size)
	for k in range(size_filter, hyp.size-size_filter-1):
		out[k] = np.max(hyp[k-size_filter:k+size_filter])
	return out

if __name__ == "__main__":
    hypnograms = get_hypnograms('train_input.csv')
    labels = get_labels('challenge_output_data_training_file_age_prediction_from_eeg_signals.csv')
    for k in range(1):
        plt.figure()
        plt.plot(hypnograms[k])
        plt.title(labels[k])
        plt.figure()
        out = max_gliss(hypnograms[k],2)
        plt.plot(out)
        plt.title(labels[k])
    plt.show()




