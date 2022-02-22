import logging
import os
import pprint

import torch.nn
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from tslearn import svm
from data_analysis import read_mat
from sklearn import metrics
import torch.optim as optim
import numpy as np


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


def svm_classification():
    # keep logs
    svm_logs_path = './svm_logs/'
    if not os.path.exists(svm_logs_path):
        os.mkdir(svm_logs_path)

    # Available features: de / psd / dasm / rasm / asm / dcau | Available smoothing: movingAve / LDS
    # features = ['de', 'asm', 'dcau']  #  'psd', 'dasm', 'rasm',
    features = ['rasm']
    smoothing = ['movingAve']  # , 'LDS'
    # smoothing = ['LDS']
    features = [f + '_' + s for f in features for s in smoothing]

    # Available bands: delta (1-3Hz)/ theta (4-7Hz)/ alpha (8-13Hz)/ beta (14-30Hz)/ gamma(31-50Hz)
    bands = {'delta': 0,
             'theta': 1,
             'alpha': 2,
             'beta': 3,
             'gamma': 4}

    # Available kernels:
    kernels = ['rbf']  # 'rbf', 'gak', 'linear', 'sigmoid'

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
                best_results_info = 'feature: ' + feature + ' band: ' + band + ' kernel: ' + kernel + ' \n' + pprint.pformat(
                    results)

            print('feature: ' + feature + ' band: ' + band + ' kernel: ' + kernel + ' \n' + pprint.pformat(results))

    # print(best_results_info)
    return best_results


class SingleFeedForwardNN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=None, last_layer=False):

        super(SingleFeedForwardNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dropout = nn.Dropout(p=dropout_rate)

        self.act = nn.LeakyReLU(negative_slope=0.2)

        self.layernorm = nn.LayerNorm(self.output_dim)

        self.last_layer = last_layer

        self.linear = nn.Linear(self.input_dim, self.output_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input):
        # Linear layer
        output = self.linear(input)

        if not self.last_layer:
            # non-linearity
            output = self.act(output)
            # dropout
            output = self.dropout(output)
            # skip connection
            output = output + input
            # layer normalization
            output = self.layernorm(output)
        else:
            output = nn.Softmax(output)

        return output


class MultiLayerFeedForwardNN(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_layers=0, dropout_rate=None, hidden_dim=-1):
        super(MultiLayerFeedForwardNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList()

        self.layers.append(SingleFeedForwardNN(input_dim=self.input_dim,
                                               output_dim=self.hidden_dim,
                                               dropout_rate=self.dropout_rate))

        for i in range(self.num_hidden_layers):
            self.layers.append(SingleFeedForwardNN(input_dim=self.hidden_dim,
                                                   output_dim=self.hidden_dim,
                                                   dropout_rate=self.dropout_rate))

        self.layers.append(SingleFeedForwardNN(input_dim=self.hidden_dim,
                                               output_dim=self.output_dim,
                                               dropout_rate=self.dropout_rate,
                                               last_layer=True))

    def forward(self, input):
        output = input
        for i in range(len(self.layers)):
            output = self.layers[i](output)

        return output


class EEGDataset(Dataset):
    def __init__(self, x):
        self.X = np.array(x)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index])


def ffn_classification():

    epochs = 10
    batch_size = 5
    nn_layers = 10
    dropout = 0.5
    hidden_dim = 512

    # keep logs
    ffn_logs_path = './ffn_logs/'
    if not os.path.exists(ffn_logs_path):
        os.mkdir(ffn_logs_path)

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

    best_results = {}
    best_results_info = ''
    band = 'gamma'
    for feature in features:
        train_x, train_y = get_data(starting_per=1, ending_per=2, feature=feature, band=bands[band])
        test_x, test_y = get_data(starting_per=2, ending_per=3, feature=feature, band=bands[band])

        train_dataset = EEGDataset(train_x)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # ffn = MultiLayerFeedForwardNN(input_dim=, output_dim=3, num_hidden_layers=nn_layers, dropout_rate=dropout, hidden_dim=hidden_dim)
        # optimizer = optim.Adam(ffn.parameters(), lr=0.01)

        # for epoch in range(epochs):
        #
        #     optimizer.zero_grad()
        #     for i, batch in enumerate(train_loader):
        #
        #         output = ffn(batch)
        #         loss = CrossEntropyLoss(output, train_y)
        #         loss.backward()
        #         optimizer.step()


if __name__ == '__main__':
    svm_classification()
    # ffn_classification()
