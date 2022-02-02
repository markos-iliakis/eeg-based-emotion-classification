import logging
import os
import pprint
from tslearn import svm
from data_analysis import read_mat
from sklearn import metrics


def get_data(starting_per, ending_per, feature, band):
    # label of each experiment (total 15) and for the entire experiment (aprox 4 minutes / 200 Hz freq) 1 -> positive_feeling / 0 -> neutral_feeling / -1 -> negative_feeling
    labels = ['1', '0', '-1', '-1', '0', '1', '-1', '0', '1', '1', '0', '-1', '0', '1', '-1']

    exp = [[], []]
    for person in range(starting_per, ending_per):

        eegs, eeg_feats = read_mat('./Data/SEED/ExtractedFeatures/', person, feature)

        for session in eeg_feats:
            for i, experiment in enumerate(session):
                experiment = experiment[:, :185, band]
                exp[0].append(experiment.tolist())
                exp[1].append(labels[i])
    return exp[0], exp[1]


if __name__ == '__main__':
    # keep logs
    svm_logs_path = './svm_logs/'
    if not os.path.exists(svm_logs_path):
        os.mkdir(svm_logs_path)

    # Available features: de / psd / dasm / rasm / asm / dcau | Available smoothing: movingAve / LDS
    features = ['de', 'psd', 'dasm', 'rasm', 'asm', 'dcau']
    smoothing = ['movingAve', 'LDS']
    features = [f + '_' + s for f in features for s in smoothing]

    # Available bands: delta (1-3Hz)/ theta (4-7Hz)/ alpha (8-13Hz)/ beta (14-30Hz)/ gamma(31-50Hz)
    bands = {'delta': 0,
             'theta': 1,
             'alpha': 2,
             'beta': 3,
             'gamma': 4}

    # Available kernels:
    kernels = ['rbf', 'gak', 'linear', 'sigmoid']

    best_results = {}
    best_results_info = ''
    band = 'gamma'
    for feature in features:
        for kernel in kernels:

            train_x, train_y = get_data(starting_per=1, ending_per=14, feature=feature, band=bands[band])
            test_x, test_y = get_data(starting_per=14, ending_per=16, feature=feature, band=bands[band])

            clf = svm.TimeSeriesSVC(kernel=kernel)
            clf.fit(train_x, train_y)
            pred_y = clf.predict(test_x)

            results = metrics.classification_report(test_y, pred_y, output_dict=True)

            if not best_results:
                best_results = results

            if results['macro avg']['f1-score'] >= best_results['macro avg']['f1-score']:
                best_results = results
                best_results_info = 'feature: ' + feature + ' band: ' + band + ' kernel: ' + kernel + ' \n' + pprint.pformat(results)

            logging.info('feature: ' + feature + ' band: ' + band + ' kernel: ' + kernel + ' \n' + pprint.pformat(results))

    print(best_results_info)
