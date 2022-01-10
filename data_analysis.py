import pyedflib
import numpy as np
from pyedflib import highlevel
from os import listdir
from os.path import isfile, join
import mne
import scipy.io
import xlrd


def read_xlsx():
    workbook = xlrd.open_workbook("./Data/SEED/channel-order.xlsx", "rb")
    sheet = workbook.sheet_by_index(0)
    rows = []
    for i in range(sheet.nrows):
        columns = []
        for j in range(sheet.ncols):
            columns.append(sheet.cell(i, j).value)
        rows.append(columns)

    print('hi')


def create_edf():
    channel_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7',
                     'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
                     'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1',
                     'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ',
                     'O2', 'CB2']

    genders = ['M', 'F', 'F', 'M', 'M', 'M', 'F', 'F', 'M', 'F', 'F', 'M', 'F', 'M', 'F']
    labels = ['1', '0', '-1', '-1', '0', '1', '-1', '0', '1', '1', '0', '-1', '0', '1', '-1']

    signals = read_mat()

    # TODO get physical min and max from actual data
    signal_headers = highlevel.make_signal_headers(channel_names, physical_min=-33000, physical_max=33000)

    # TODO check FILETYPE_BDFPLUS
    for human_i, (gender, signal) in enumerate(zip(genders, signals)):
        for i, experiment in enumerate(signal):
            header = highlevel.make_header(patientname=str(human_i) + '_exp' + str(i), gender=gender)
            highlevel.write_edf('./Data/SEED/edfs/' + str(human_i) + '_exp' + str(i) + '.edf', experiment, signal_headers, header, file_type=pyedflib.FILETYPE_EDFPLUS)
            f = pyedflib.EdfWriter('./Data/SEED/edfs/' + str(human_i) + '_exp' + str(i) + '.edf', 1, file_type=pyedflib.FILETYPE_EDFPLUS)
            f.writeAnnotation(0, 240, labels[i], str_format='utf-8')
            break
        break

    print('hi')
    # mne.find_events(bdf_raw)
    # n = f.signals_in_file
    # signal_labels = f.getSignalLabels()
    # sigbufs = np.zeros((n, f.getNSamples()[0]))
    # for i in np.arange(n):
    #     sigbufs[i, :] = f.readSignal(i)
    #
    # print('finished')

    # d = {}
    # for i in range(256):
    #     d[f.getLabel(i)] = f.getLabel(i)[1:]
    #
    # f.close()
    # print("hiiii")
    #
    # signals, signal_headers, header = highlevel.read_edf('./Data/eeg_recording_1.bdf')
    # highlevel.drop_channels('./Data/eeg_recording_1.bdf',
    #                         to_drop=['Ana1-2', 'Ana3-4', 'Ana5-6', 'Ana7-8', 'Ana9-10', 'Ana11-12', 'Ana13-14',
    #                                  'Ana15-16', 'Status'])
    # highlevel.rename_channels('./Data/eeg_recording_1_original.bdf', mapping=d)

    # print('hi')


def read_mat():
    # Get file names
    onlyfiles = [f for f in listdir('./Data/SEED/Preprocessed_EEG/') if isfile(join('./Data/SEED/Preprocessed_EEG/', f))]
    human_eegs = list()

    # Read files
    for file in onlyfiles:
        mat3 = scipy.io.loadmat('./Data/SEED/Preprocessed_EEG/' + file)
        human_eegs.append(list([value for key, value in mat3.items() if 'eeg' in key.lower()]))
        break

    return human_eegs


def read_edf():
    onlyfiles = [f for f in listdir('./Data/SEED/edfs/') if isfile(join('./Data/SEED/edfs/', f))]
    for file in onlyfiles:
        # f = pyedflib.EdfReader('./Data/SEED/edfs/' + file, filetype=pyedflib.FILETYPE_EDFPLUS)
        # x = f.readAnnotations()
        x = highlevel.read_edf('./Data/SEED/edfs/' + file)
    print('hi')


if __name__ == '__main__':
    # create_edf()
    read_edf()
