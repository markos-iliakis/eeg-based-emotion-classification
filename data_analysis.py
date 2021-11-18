import pyedflib
import numpy as np
from pyedflib import highlevel

if __name__ == '__main__':

    file_name = pyedflib.data.get_generator_filename()
    file_name = './Data/eeg_recording_1.bdf'
    f = pyedflib.EdfReader(file_name)
    # n = f.signals_in_file
    # signal_labels = f.getSignalLabels()
    # sigbufs = np.zeros((n, f.getNSamples()[0]))
    # for i in np.arange(n):
    #     sigbufs[i, :] = f.readSignal(i)
    #
    # print('finished')

    d = {}
    for i in range(256):
        d[f.getLabel(i)] = f.getLabel(i)[1:]

    f.close()
    # print("hiiii")
    #
    # signals, signal_headers, header = highlevel.read_edf('./Data/eeg_recording_1.bdf')
    # highlevel.drop_channels('./Data/eeg_recording_1.bdf',
    #                         to_drop=['Ana1-2', 'Ana3-4', 'Ana5-6', 'Ana7-8', 'Ana9-10', 'Ana11-12', 'Ana13-14',
    #                                  'Ana15-16', 'Status'])
    highlevel.rename_channels('./Data/eeg_recording_1_original.bdf', mapping=d)

    # print('hi')