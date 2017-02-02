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
	plt.scatter(list_points, 2*np.ones(len(list_points)), color='r')
	return out

def hyp_to_list_nrem_cycle_and_plot(hyp):
	out = []
	list_points = []
	id_begin = 0
	list_points.append(id_begin)
	for k in range(1,hyp.size):
		if hyp[k]==2 and hyp[k-1]!=2 and hyp[k-1]!=3:
			out.append(hyp[id_begin:k-1])
			id_begin = k
			list_points.append(id_begin)
	plt.figure()
	plt.plot(hyp)
	plt.scatter(list_points, 2*np.ones(len(list_points)), color='g')
	return out

def hyp_to_list_rem_cycle_and_plot(hyp):
	out = []
	list_points = []
	id_begin = 0
	list_points.append(id_begin)
	for k in range(1,hyp.size):
		if hyp[k]==4 and hyp[k-1]!=4:
			out.append(hyp[id_begin:k-1])
			id_begin = k
			list_points.append(id_begin)
	plt.figure()
	plt.plot(hyp)
	plt.scatter(list_points, 2*np.ones(len(list_points)), color='y')
	return out	

def hyp_to_periods(hyp):
	out = np.zeros(hyp.size)
	k = 0
	out[k] = 0
	while k != 2:
		k = k + 1
		out[k] = 0
	NREM = 1
	i = k + 1
	while i < hyp.size:
		if NREM == 1:
			if hyp[i] != 4:
				out[i] = 2
				i = i + 1
			else:
				NREM = 0
				out[i] = 4
				i = i + 1
		else:
			if hyp[i] == 4:
				out[i] = 4
				i = i + 1
			else:
				j = i
				while j < hyp.size and hyp[j] != 4 and j-i < 60:
					j = j + 1
				if j-i >= 59:
					NREM = 1
					out[i] = 2
					i=i+1
				else:
					out[i] = 4
					i=i+1
	return out

def hash_cycle(list_cycle):
	out = []
	for k in range(len(list_cycle)):
		out.append(is_cycle_complete(list_cycle[k]))
	return out

def feature_cycle(hyp):
	hyp_periods = hyp_to_periods(hyp)
	nrem_cycles_duration = []
	rem_cycles_duration = []
	i=0
	while hyp_periods[i] == 0:
			i=i+1

	k = i + 1
	ind = k
	while k < hyp.size  and len(rem_cycles_duration) < 4 and len(nrem_cycles_duration) < 4:
		if hyp[k]!=hyp[k-1]:
			if hyp[k] == 2:
				rem_cycles_duration.append(k-ind)
				ind = k
				k = k + 1
			else :
				nrem_cycles_duration.append(k-ind)
				ind = k
				k = k + 1
		else:
			k = k + 1

	if hyp[-1]==2 and len(nrem_cycles_duration) < 4:
		nrem_cycles_duration.append(hyp.size-ind)
	if hyp[-1]==4 and len(rem_cycles_duration) < 4:
		rem_cycles_duration.append(hyp.size-ind)

	while len(nrem_cycles_duration) < 4:
		nrem_cycles_duration.append(-1)

	while len(rem_cycles_duration) < 4:
		rem_cycles_duration.append(-1)

	return rem_cycles_duration, nrem_cycles_duration

def get_feature_cycle(hyps):
	out = []
	for hyp in hyps:
		rem_cycles_duration, nrem_cycles_duration = feature_cycle(hyp)
		out.append(np.array(rem_cycles_duration + nrem_cycles_duration))
	return np.array(out)

if __name__ == "__main__":
	hypnograms = get_hypnograms('train_input.csv')
	labels = get_labels('challenge_output_data_training_file_age_prediction_from_eeg_signals.csv')
	# for i in range(10):
	# 	#plt.figure()
	# 	#plt.plot(hypnograms[0])
	# 	#out = max_gliss(hypnograms[0],2)
	# 	#list_cycle = hyp_to_list_cycle_and_plot(out)
	# 	#list_cycle = hyp_to_list_nrem_cycle_and_plot(out)
	# 	#list_cycle = hyp_to_list_rem_cycle_and_plot(out)
	# 	hyp_periods = hyp_to_periods(hypnograms[i])
	# 	plt.figure()
	# 	plt.title(labels[i])
	# 	rem_cycles_duration, nrem_cycles_duration = feature_cycle(hyp_periods)
	# 	print(rem_cycles_duration)
	# 	print(nrem_cycles_duration)
	# 	print()
	# 	plt.plot(hyp_periods)

	plt.figure()
	plt.plot(hypnograms[2])
	hyp_periods = hyp_to_periods(hypnograms[2])
	plt.figure()
	plt.plot(hyp_periods)
	rem_cycles_duration, nrem_cycles_duration = feature_cycle(hyp_periods)
	print(rem_cycles_duration)
	print(nrem_cycles_duration)
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



