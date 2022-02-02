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
        print('File already exists. No overwrite. SKIPPING..')
        return
        # os.remove(fname)

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
        print(fname + 'CREATED')
    return True


def create_paths(edfs_path, person, sess_i, exp_i):
    # Create paths
    person_path = edfs_path + 'person' + str(person) + '/'
    # session_path = person_path + 'session' + str(sess_i) + '/'
    fname = person_path + 'pers_' + str(person) + '_sess_' + str(sess_i) + '_exp_' + str(exp_i) + '.edf'

    if not os.path.exists(person_path):
        os.mkdir(person_path)
    # if not os.path.exists(session_path):
    #     os.mkdir(session_path)

    return fname


def create_edf(mat_path, edfs_path):
    channel_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7',
                     'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
                     'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1',
                     'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ',
                     'O2', 'CB2']

    # gender of each people_exps[i]
    genders = ['M', 'F', 'F', 'M', 'M', 'M', 'F', 'F', 'M', 'F', 'F', 'M', 'F', 'M', 'F']

    # label of each experiment (total 15) and for the entire experiment (aprox 4 minutes / 200 Hz freq) 1 -> positive_feeling / 0 -> neutral_feeling / -1 -> negative_feeling
    labels = ['1', '0', '-1', '-1', '0', '1', '-1', '0', '1', '1', '0', '-1', '0', '1', '-1']

    label_names = {
        '1': "Positive",
        '0': "Neutral",
        "-1": "Negative"
    }

    # start_stop = [[0:06:13, 0:10:11], [0:00:50, 0:04:36], [0:20:09, 0:23:35], [0:49:57, 0:53:59], [0:10:39, 0:13:43], [1:05:09, 1:08:28], [2:01:20, 2:05:21], [2:55, 6:35], [1:18:56, 1:23:22], [11:31, 15:32], [10:40, 14:38], [2:16:37, 2:20:36], [5:36, 9:36], [35:00, 39:02], [1:48:52, 1:52:18]]
    info = mne.create_info(channel_names, 200, ch_types='eeg')

    # read the mat files 3 by 3 for memory efficiency
    min_eeg = 48000
    for person in range(1, 16):
        # person_sessions = 3 * number_of_experiments * (channels * time_steps)
        person_sessions, eeg_feats = read_mat(mat_path, person)
        for sess_i, session in enumerate(person_sessions):
            for exp_i, experiment in enumerate(session):
                raw_array = mne.io.RawArray(experiment, info)

                # create annotations https://mne.tools/dev/auto_tutorials/raw/30_annotate_raw.html https://mne.tools/dev/auto_tutorials/intro/20_events_from_raw.html
                my_annot = mne.Annotations(onset=[0],  # in seconds
                                           duration=[round(raw_array.last_samp/200)],  # in seconds, too
                                           description=[label_names[labels[exp_i]]])
                raw_array.set_annotations(my_annot)

                # calculate the minimum eeg
                if min_eeg > raw_array.last_samp:
                    min_eeg = raw_array.last_samp

                fname = create_paths(edfs_path=edfs_path, person=person, sess_i=sess_i, exp_i=exp_i)
                write_mne_edf(mne_raw=raw_array, fname=fname, ch_names=channel_names, overwrite=False)

    print('Minimum EEG found: ' + str(min_eeg)) #37000


def read_mat(mat_path, person_n, feature=None):
    # Get file names
    onlyfiles = [f for f in listdir(mat_path) if isfile(join(mat_path, f))]
    human_eegs = list()
    eeg_features = list()

    # Read files
    for i, file in enumerate(onlyfiles):
        if file.startswith(str(person_n) + '_'):
            eeg = scipy.io.loadmat(mat_path + file)
            human_eegs.append(list([value for key, value in eeg.items() if 'eeg' in key.lower()]))
            if feature:
                eeg_features.append(list([value for key, value in eeg.items() if key.startswith(feature)]))
            # print('Reading ' + mat_path + file)

    return human_eegs, eeg_features


def read_edf(edfs_path, failed_edfs_path):
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(edfs_path):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]


    # onlyfiles = [f for f in listdir('./Data/SEED/edfs/') if isfile(join('./Data/SEED/edfs/', f))]
    for file in listOfFiles:
        try:
            x = mne.io.read_raw_edf(file, preload=True)

        except Exception:
            os.rename(file,  failed_edfs_path + os.path.basename(file))
            print('Error in file: ' + file + ' . MOVED to ' + failed_edfs_path)


if __name__ == '__main__':
    mat_path = './Data/SEED/Preprocessed_EEG/'
    edfs_path = './Data/SEED/edfs/'
    failed_edfs_path = './Data/SEED/failed_edfs/'
    # create_edf(mat_path=mat_path, edfs_path=edfs_path)
    # read_edf(edfs_path=edfs_path, failed_edfs_path=failed_edfs_path)
    read_mat('./Data/SEED/ExtractedFeatures/', 2)
