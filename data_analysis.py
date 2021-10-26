import pyedflib
import numpy as np

if __name__ == '__main__':

    # file_name = pyedflib.data.get_generator_filename()
    file_name = './Data/eeg_recording_1.bdf'
    f = pyedflib.EdfReader(file_name)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)
