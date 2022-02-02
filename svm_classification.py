from sklearn import svm
from tslearn import svm
from data_analysis import read_mat

if __name__ == '__main__':
    # label of each experiment (total 15) and for the entire experiment (aprox 4 minutes / 200 Hz freq) 1 -> positive_feeling / 0 -> neutral_feeling / -1 -> negative_feeling
    labels = ['1', '0', '-1', '-1', '0', '1', '-1', '0', '1', '1', '0', '-1', '0', '1', '-1']

    # Available features: de / psd / dasm / rasm / asm / dcau | Available smoothing: movingAve / LDS
    eegs, eeg_feats = read_mat('./Data/SEED/ExtractedFeatures/', 2, 'de_movingAve')

    clf = svm.TimeSeriesSVC()
    clf.fit()