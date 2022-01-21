import os
from datetime import timedelta, datetime, timezone

import pyedflib
import numpy as np
from pyedflib import highlevel, FILETYPE_EDFPLUS, FILETYPE_EDF, FILETYPE_BDFPLUS, FILETYPE_BDF
from os import listdir
from os.path import isfile, join
import mne
import scipy.io
import xlrd

# Each SEED/preprocessed_EEG/*.mat file contains each person's (total 15) signals from each experiment (total 15)
# read_mat -> to take the signals and hardcoded channel_names, genders and labels taken from xlsx to create the headers


def _stamp_to_dt(utc_stamp):
    """Convert timestamp to datetime object in Windows-friendly way."""
    if 'datetime' in str(type(utc_stamp)): return utc_stamp
    # The min on windows is 86400
    stamp = [int(s) for s in utc_stamp]
    if len(stamp) == 1:  # In case there is no microseconds information
        stamp.append(0)
    return (datetime.fromtimestamp(0, tz=timezone.utc) +
            timedelta(0, stamp[0], stamp[1]))  # day, sec, μs


def write_mne_edf(mne_raw, fname, ch_names, picks=None, tmin=0, tmax=None,
                  overwrite=False):
    """
    Saves the raw content of an MNE.io.Raw and its subclasses to
    a file using the EDF+/BDF filetype
    pyEDFlib is used to save the raw contents of the RawArray to disk
    Parameters
    ----------
    mne_raw : mne.io.Raw
        An object with super class mne.io.Raw that contains the data
        to save
    fname : string
        File name of the new dataset. This has to be a new filename
        unless data have been preloaded. Filenames should end with .edf
    picks : array-like of int | None
        Indices of channels to include. If None all channels are kept.
    tmin : float | None
        Time in seconds of first sample to save. If None first sample
        is used.
    tmax : float | None
        Time in seconds of last sample to save. If None last sample
        is used.
    overwrite : bool
        If True, the destination file (if it exists) will be overwritten.
        If False (default), an error will be raised if the file exists.
    """
    if not issubclass(type(mne_raw), mne.io.BaseRaw):
        raise TypeError('Must be mne.io.Raw type')
    if not overwrite and os.path.exists(fname):
        # raise OSError('File already exists. No overwrite.')
        os.remove(fname)

    # static settings
    has_annotations = True if len(mne_raw.annotations) > 0 else False
    if os.path.splitext(fname)[-1] == '.edf':
        file_type = FILETYPE_EDFPLUS if has_annotations else FILETYPE_EDF
        dmin, dmax = -32768, 32767
    else:
        file_type = FILETYPE_BDFPLUS if has_annotations else FILETYPE_BDF
        dmin, dmax = -8388608, 8388607

    print('saving to {}, filetype {}'.format(fname, file_type))
    sfreq = mne_raw.info['sfreq']
    # date = _stamp_to_dt(mne_raw.info['meas_date'])

    # if tmin:
    #     date += timedelta(seconds=tmin)
    # no conversion necessary, as pyedflib can handle datetime.
    # date = date.strftime('%d %b %Y %H:%M:%S')
    first_sample = int(sfreq * tmin)
    last_sample = int(sfreq * tmax) if tmax is not None else None

    # convert data
    channels = mne_raw.get_data(picks,
                                start=first_sample,
                                stop=last_sample)

    # convert to microvolts to scale up precision
    channels *= 1e6

    # set conversion parameters
    n_channels = len(channels)

    # create channel from this
    try:
        f = pyedflib.EdfWriter(fname,
                               n_channels=n_channels,
                               file_type=file_type)

        channel_info = []

        ch_idx = range(n_channels) if picks is None else picks
        # keys = list(mne_raw._orig_units.keys())
        for i in ch_idx:
            try:
                ch_dict = {'label': mne_raw.ch_names[i],
                           'dimension': 'mv',
                           'sample_rate': mne_raw._raw_extras[0]['n_samps'][i],
                           'physical_min': mne_raw._raw_extras[0]['physical_min'][i],
                           'physical_max': mne_raw._raw_extras[0]['physical_max'][i],
                           'digital_min': mne_raw._raw_extras[0]['digital_min'][i],
                           'digital_max': mne_raw._raw_extras[0]['digital_max'][i],
                           'transducer': '',
                           'prefilter': ''}
            except:
                ch_dict = {'label': mne_raw.ch_names[i],
                           'dimension': 'mv',
                           'sample_rate': sfreq,
                           'physical_min': channels.min(),
                           'physical_max': channels.max(),
                           'digital_min': dmin,
                           'digital_max': dmax,
                           'transducer': '',
                           'prefilter': ''}

            channel_info.append(ch_dict)
        f.setTechnician('mne-gist-save-edf-skjerns')
        f.setSignalHeaders(channel_info)
        for i in range(n_channels):
            f.setLabel(i, label=ch_names[i])

        f.writeSamples(channels)

        for annotation in mne_raw.annotations:
            description = annotation['description']
            onset = annotation['onset']
            duration = annotation['duration']
            f.writeAnnotation(onset, duration, description)

    except Exception as e:
        raise e
    finally:
        f.close()
    return True


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

    # gender of each signals[i]
    genders = ['M', 'F', 'F', 'M', 'M', 'M', 'F', 'F', 'M', 'F', 'F', 'M', 'F', 'M', 'F']

    # label of each experiment (total 15) and for the entire experiment (aprox 4 minutes / 200 Hz freq) 1 -> positive_feeling / 0 -> neutral_feeling / -1 -> negative_feeling
    labels = ['1', '0', '-1', '-1', '0', '1', '-1', '0', '1', '1', '0', '-1', '0', '1', '-1']

    # signals = number_of_people * number_of_experiments * (channels * time_steps)
    signals = read_mat()

    # start_stop = [[0:06:13, 0:10:11], [0:00:50, 0:04:36], [0:20:09, 0:23:35], [0:49:57, 0:53:59], [0:10:39, 0:13:43], [1:05:09, 1:08:28], [2:01:20, 2:05:21], [2:55, 6:35], [1:18:56, 1:23:22], [11:31, 15:32], [10:40, 14:38], [2:16:37, 2:20:36], [5:36, 9:36], [35:00, 39:02], [1:48:52, 1:52:18]]
    info = mne.create_info(channel_names, 200, ch_types='eeg')
    raw_array = mne.io.RawArray(signals[0][0], info)

    # create annotations https://mne.tools/dev/auto_tutorials/raw/30_annotate_raw.html https://mne.tools/dev/auto_tutorials/intro/20_events_from_raw.html
    my_annot = mne.Annotations(onset=[0],  # in seconds
                               duration=[240],  # in seconds, too
                               description=['Positive'])
    raw_array.set_annotations(my_annot)

    # stim channel
    # stim_info = mne.create_info(['STI 014'], raw_array.info['sfreq'], ['stim'])
    # stim_raw = mne.io.RawArray(np.zeros((1, len(raw_array.times))), stim_info)
    # raw_array.add_channels([stim_raw], force_update_info=True)

    write_mne_edf(raw_array, fname='./Data/SEED/edfs/' + str(1) + '_exp' + str(1) + '.edf', ch_names=channel_names)


def read_mat():
    # Get file names
    onlyfiles = [f for f in listdir('./Data/SEED/Preprocessed_EEG/') if isfile(join('./Data/SEED/Preprocessed_EEG/', f))]
    human_eegs = list()

    # Read files
    for file in onlyfiles:
        mat3 = scipy.io.loadmat('./Data/SEED/Preprocessed_EEG/' + file)
        # mat3 = scipy.io.loadmat('./Data/SEED/Preprocessed_EEG/2_20140419.mat')
        human_eegs.append(list([value for key, value in mat3.items() if 'eeg' in key.lower()]))
        break

    return human_eegs


def read_edf():
    onlyfiles = [f for f in listdir('./Data/SEED/edfs/') if isfile(join('./Data/SEED/edfs/', f))]
    for file in onlyfiles:
        x = mne.io.read_raw_edf('./Data/SEED/edfs/' + file, preload=True)

    print('hi')


if __name__ == '__main__':
    create_edf()
    read_edf()
