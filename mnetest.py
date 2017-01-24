import numpy as np
import mne
import matplotlib.pyplot as plt
from parserythm import *
from mne.preprocessing import create_ecg_epochs, create_eog_epochs

if __name__ == "__main__":
    train_hyp, train_eegs, train_devices, train_labels, eval_hyp, eval_eegs, eval_devices, eval_labels = train_eval_base()
    ch_types = ['eeg']
    ch_name = ['train0']
    sfreq = 250
    info = mne.create_info(ch_names=ch_name, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eval_eegs[0][None, :], info)
    raw2 = mne.io.RawArray(eval_eegs[1][None, :], info)
    scalings = 'auto'
    raw.plot(n_channels=1, show=True, scalings=scalings)
    raw.plot_psd(fmax = 250)
    average_ecg = create_ecg_epochs(raw).average()
    print('We found {} ECG events'.format(average_ecg.nave))
    plt.show()

