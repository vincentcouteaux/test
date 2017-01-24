import numpy as np
import pywt
import scipy as sp
import scipy.signal
from parserythm import *

FE = 250

def wavelet(eeg, age):
    coefs, freq = pywt.cwt(eeg, np.arange(1, 129), 'shan')
    plt.figure()
    plt.imshow(np.flipud(np.log(np.abs(coefs))), aspect="auto")
    plt.title('age = {}, max freq = {}'.format(age, freq[-1]))

def stft(eeg, age):
    plt.figure()
    f, t, s = sp.signal.spectrogram(eeg, fs=250., nperseg=256)
    s=np.flipud(s)
    plt.imshow(np.log(s), aspect="auto", extent=[t[0], t[-1],f[0], f[-1]])
    plt.title('age = {}'.format(age))

def fft(eeg, age):
    plt.figure()
    spectrum = np.log(np.abs(np.fft.fft(eeg)))
    spectrum = spectrum[:(spectrum.size/2)]
    plt.plot(spectrum)
    plt.title('age: {} y.o'.format(int(age)))

def f_0(eeg):
    spectrum = np.log(np.abs(np.fft.fft(eeg)))
    maxbin = 4.*eeg.size/FE
    spectrum = spectrum[:maxbin]
    fbin = np.argmax(spectrum)
    return float(fbin)/eeg.size*FE

def f_0s(eegs):
    out = []
    for eeg in eegs:
        out.append(f_0(eeg))
    return out

def spectrums(eegs, f_max, f_min = -1, n=-1):
    if n == -1:
        n = eegs.shape[1]/2
    if f_min == -1:
        min_bin = 0
    else:
        min_bin = int(float(f_min)/FE*n)
    s = np.abs(np.fft.fft(eegs, n=n))
    max_bin = int(float(f_max)/FE*n)
    return s[:, min_bin:max_bin]

def maxf (eegs):
    spec = spectrums(eegs, 10.)
    #print(np.argmax(spec, 1))
    return np.max(spec, 1)

def max_alpha(eegs):
    spec=spectrums(eegs, 30., 10., 1024)
    #print(np.argmax(spec, 1)*float(FE)/eegs.shape[1])
    for k in range(10):
        plt.plot(spec[k])
        plt.show()
    return np.max(spec, 1)

def above75(eegs):
    return np.sum(np.abs(eegs) > 75., 1)


def all_ages_stft(eegs, ages):
    for age in range(int(np.min(ages)), int(np.max(ages)+1)):
        if age in ages:
            ind = np.where(ages == age)[0][0]
            stft(eegs[ind], age)
            #fft(eegs[ind], age)
            #wavelet(eegs[ind], age)


if __name__ == "__main__":
    eegs = get_eegs('train_input.csv')
    labels = get_labels('challenge_output_data_training_file_age_prediction_from_eeg_signals.csv')
    devices = get_device('train_input.csv')
#    stft(eegs[0], labels[0])
    #stft(eegs[1], labels[1])
#    stft(eegs[2], labels[2])
#    stft(eegs[np.argmax(labels)], np.max(labels))
#    stft(eegs[np.argmin(labels)], np.min(labels))
    #plt.hist(labels, 40)
    #all_ages_stft(eegs, labels)
    #wavelet(eegs[2], labels[2])
    plt.scatter(labels[devices==0], above75(eegs[devices==0]))
    plt.figure()
    plt.scatter(labels[devices==1], above75(eegs[devices==1]))
    plt.show()

