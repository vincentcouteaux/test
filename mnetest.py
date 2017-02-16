import numpy as np
import mne
import matplotlib.pyplot as plt
from parserythm import *
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
from mne.time_frequency import psd_welch, tfr_morlet

def get_psds(eegs):
    out=[]
    ch_types = ['eeg']
    ch_name = ['train0']
    sfreq = 250
    info = mne.create_info(ch_names=ch_name, sfreq=sfreq, ch_types=ch_types)
    for eeg in eegs:
        raw = mne.io.RawArray(eeg[None, :], info)
        psd = psd_welch(raw)
        out.append(np.log(psd[0][0,:]))
        freqs = psd[1]
    return out, freqs

def max_pool(psd, fs, freqs, max_freq, f_gap):
    out = np.zeros(int(max_freq/f_gap))
    for k in range(int(max_freq/f_gap)):
        beg = int(k*float(f_gap)/fs*psd.size*2)
        end = int((k+1)*float(f_gap)/fs*psd.size*2)
        out[k] = np.max(psd[beg:end])
    return out

def max_pools(eegs, max_freq, f_gap):
    psds, freqs = get_psds(eegs)
    out = []
    for psd in psds:
        out.append(max_pool(psd, 250, freqs, max_freq, f_gap))
    return np.array(out)

def get_wavelet(eegs):
    out=[]
    ch_types = ['eeg']
    ch_name = ['train0']
    sfreq = 250
    info = mne.create_info(ch_names=ch_name, sfreq=sfreq, ch_types=ch_types)
    freqs = np.arange(2., 40., 2.)
    ncycles = freqs/2
    for eeg in eegs:
        raw = mne.io.RawArray(eeg[None, :], info)
        wv = tfr_morlet(raw, freqs, ncycles) #Doesn't work: needs epochs
        wv.plot([0])
        #out.append(np.log(psd[0][0,:]))
    return out, freqs


if __name__ == "__main__":
    train_hyp, train_eegs, train_devices, train_labels, eval_hyp, eval_eegs, eval_devices, eval_labels = train_eval_base()
    #psds, freqs = get_psds(eval_eegs)
    #get_psds(train_eegs)
    #pools = max_pools(eval_eegs, 40, 5)
    ch_types = ['eeg']
    ch_name = ['train0']
    sfreq = 250
    info = mne.create_info(ch_names=ch_name, sfreq=sfreq, ch_types=ch_types)
    eeg = train_eegs[15]
    raw = mne.io.RawArray(eeg[None, :], info)
    print(train_labels[15])
    raw.plot(show=True)
    eeg = train_eegs[16]
    raw = mne.io.RawArray(eeg[None, :], info)
    raw.plot(show=True)
    print(train_labels[16])
    #for pool in pools[:15]:
    #    plt.figure()
    #    plt.plot(pool)
    #plt.show()




