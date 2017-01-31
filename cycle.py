import numpy as np
import matplotlib.pyplot as plt
from parserythm import *

def max_gliss(hyp, size_filter):
	out = np.zeros(hyp.size)
	for k in range(size_filter, hyp.size-size_filter-1):
		out[k] = np.max(hyp[k-size_filter:k+size_filter])
	return out

def list_max_gliss(hyps, size_filter):
	out = []
	for hyp in hyps:
		out.append(max_gliss(hyp, size_filter)) 
	return out

def nb_cycle(hyp):
	nb = 0
	for k in range(1,hyp.size):
		if hyp[k]==0 and hyp[k-1]!=0:
			nb+=1
	return nb

def hyp_to_list_cycle_and_plot(hyp):
	out = []
	list_points = []
	id_begin = 0
	list_points.append(id_begin)
	for k in range(1,hyp.size):
		if hyp[k]==0 and hyp[k-1]!=0:
			out.append(hyp[id_begin:k-1])
			id_begin = k
			list_points.append(id_begin)
	plt.figure()
	plt.plot(hyp)
	plt.scatter(list_points, 2*np.ones(len(list_points)), color = 'r')
	return out

def hash_cycle(cycle):
	out = []







if __name__ == "__main__":
	hypnograms = get_hypnograms('train_input.csv')
	labels = get_labels('challenge_output_data_training_file_age_prediction_from_eeg_signals.csv')
	plt.figure()
	plt.plot(hypnograms[0])
	out = max_gliss(hypnograms[0],1)
	list_cycle = hyp_to_list_cycle_and_plot(out)
	plt.show()
#    for k in range(1):
#        plt.figure()
#        plt.plot(hypnograms[k])
#        plt.title(labels[k])
#        plt.figure()
#        out = max_gliss(hypnograms[k],2)
#        plt.plot(out)
#        plt.title(labels[k])
#    plt.show()



