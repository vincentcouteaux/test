import numpy as np
import matplotlib.pyplot as plt
import csv
import sys

csv.field_size_limit(sys.maxsize)

def get_eegs(filename):
    eegs = np.zeros((581, 75000))
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        c = 0
        reader.next()
        for row in reader:
            eegs[c, :] = row[2:75002]
            c += 1
    return eegs

def hypnogram_to_list(v):
    out = []
    cpt = 0
    for c in v:
        if c == '0':
            out.append(0)
        elif c == '1':
            if v[cpt-1] == '-':
                out.append(-1)
            else:
                out.append(1)
        elif c == '2':
            out.append(2)
        elif c == '3':
            out.append(3)
        elif c == '4':
            out.append(4)
        cpt += 1
    return out


def get_hypnograms(filename):
    hypnograms = []
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        reader.next()
        for row in reader:
            hypnograms.append(np.array(hypnogram_to_list(row[75002])))
    return hypnograms

def train_eval_base():
    hypnograms = []
    eegs = np.zeros((581, 75000))
    filename = 'train_input.csv'
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        reader.next()
        c = 0
        for row in reader:
            hypnograms.append(np.array(hypnogram_to_list(row[75002])))
            eegs[c, :] = row[2:75002]
            c += 1
    labels = get_labels('challenge_output_data_training_file_age_prediction_from_eeg_signals.csv')
    devices = get_device('train_input.csv')
    return hypnograms[:500], eegs[:500], devices[:500], labels[:500], hypnograms[500:], eegs[500:], devices[500:], labels[500:]

def test_base():
    hypnograms = []
    eegs = np.zeros((249, 75000))
    filename = 'test_input.csv'
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        reader.next()
        c = 0
        for row in reader:
            hypnograms.append(np.array(hypnogram_to_list(row[75002])))
            eegs[c, :] = row[2:75002]
            c += 1
    return hypnograms, eegs


def get_labels(filename):
    labels = np.zeros(581)
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        reader.next()
        c = 0
        for row in reader:
            labels[c] = int(row[1])
            c += 1
    return labels

def get_device(filename):
    devices = []
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        reader.next()
        for row in reader:
            devices.append(int(float(row[1])))
    return np.array(devices)

if __name__ == "__main__":
    print(hypnogram_to_list("['1', '2', '-1', '-1', '4']"))
    eegs = get_eegs('train_input.csv')
    hypnograms = get_hypnograms('train_input.csv')
    labels = get_labels('challenge_output_data_training_file_age_prediction_from_eeg_signals.csv')
    devices = get_device('train_input.csv')
    #plt.plot(eegs[1, :])
    #plt.plot(eegs[2, :])
    #plt.plot(eegs[3, :])
    for k in range(2):
        plt.figure()
        plt.plot(hypnograms[k+15])
        plt.title('hypnogram of a {} y.o'.format(labels[k+15]))
        plt.xlabel('Time')
        plt.ylabel('Sleep stage')
        plt.figure()
        plt.plot(eegs[k+15])
        plt.title('electroencephalogram of a {} y.o'.format(labels[k+15]))
        plt.xlabel('Time (sample at 250Hz)')
        plt.ylabel('Voltage unit')
    plt.figure()
    plt.hist(labels, bins=int(np.max(labels)-np.min(labels)))
    plt.title('histogram of ages (train base)')
    plt.xlabel('age')
    plt.ylabel('count')
    print('mean ages = {}'.format(np.mean(labels)))
    print('std ages = {}'.format(np.std(labels)))
    plt.show()

