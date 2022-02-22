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
    labels = [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]

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

            train_x, train_y = get_data(starting_per=1, ending_per=8, feature=feature, band=bands[band])
            test_x, test_y = get_data(starting_per=8, ending_per=10, feature=feature, band=bands[band])

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

            logging.info(
                'feature: ' + feature + ' band: ' + band + ' kernel: ' + kernel + ' \n' + pprint.pformat(results))

    print(best_results_info)
    return best_results


class SingleFeedForwardNN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=None, last_layer=False, first_layer=False):

        super(SingleFeedForwardNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dropout = nn.Dropout(p=dropout_rate)

        self.act = nn.LeakyReLU(negative_slope=0.2)

        self.layernorm = nn.LayerNorm(self.output_dim)

        self.softmax = nn.Softmax(dim=1)

        self.last_layer = last_layer
        self.first_layer = first_layer

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
            if not self.first_layer:
                output = output + input
            # layer normalization
            output = self.layernorm(output)
        else:
            output = self.softmax(output)

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
                                               dropout_rate=self.dropout_rate,
                                               first_layer=True))

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
    def __init__(self, x, y):
        self.X = np.array(x)
        self.Y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index].astype(np.float32)), torch.from_numpy(np.array(self.Y[index]))


def ffn_classification():
    epochs = 200
    batch_size = 5
    nn_layers = 0
    dropout = 0.5
    hidden_dim = 512
    lr = 0.1

    # keep logs
    ffn_logs_path = './ffn_logs/'
    if not os.path.exists(ffn_logs_path):
        os.mkdir(ffn_logs_path)

    # Available features: de / psd / dasm / rasm / asm / dcau | Available smoothing: movingAve / LDS
    # features = ['de', 'psd', 'dasm', 'rasm', 'asm', 'dcau']
    features = ['de', 'asm', 'dcau']
    # smoothing = ['movingAve', 'LDS']
    smoothing = ['LDS']
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
        print("Feature ", feature, "..............................................")
        train_x, train_y = get_data(starting_per=1, ending_per=11, feature=feature, band=bands[band])
        test_x, test_y = get_data(starting_per=11, ending_per=15, feature=feature, band=bands[band])

        channels = len(train_x[0])
        sec = len(train_x[0][0])
        train_dataset = EEGDataset(train_x, train_y)
        test_dataset = EEGDataset(test_x, test_y)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        ffn = MultiLayerFeedForwardNN(input_dim=channels*sec, output_dim=3, num_hidden_layers=nn_layers, dropout_rate=dropout,
                                      hidden_dim=hidden_dim)

        criterion = CrossEntropyLoss()

        optimizer = optim.Adam(ffn.parameters(), lr=lr)

        for epoch in range(epochs):
            print("Epoch ", epoch, "..................")
            optimizer.zero_grad()
            # Train
            for train_batch, train_batch_labels in train_loader:
                train_batch = train_batch.view(-1, channels*sec).requires_grad_()
                output = ffn(train_batch)
                loss = criterion(output, train_batch_labels)
                loss.backward()
                optimizer.step()

            # Calc metrics for test
            test_pred = []
            test_labels = []
            for test_batch, test_batch_labels in test_loader:

                test_batch = test_batch.view(-1, channels*sec).requires_grad_()
                output = ffn(test_batch)

                _, predicted = torch.max(output, 1)

                test_labels.extend(test_batch_labels.tolist())
                test_pred.extend(predicted.tolist())

            results = metrics.classification_report(test_labels, test_pred, output_dict=True, zero_division=0)

            if not best_results:
                best_results = results

            if results['macro avg']['f1-score'] >= best_results['macro avg']['f1-score']:
                best_results = results
                best_results_info = 'feature: ' + feature + ' band: ' + band + ' epoch: ' + str(epoch) + ' lr: ' + str(lr) + ' nn_layers: ' + str(nn_layers) +  ' \n' + pprint.pformat(
                    results)

            pprint.pformat(results)
            logging.info(
                'feature: ' + feature + ' band: ' + band + ' epoch: ' + str(epoch) + ' lr: ' + str(lr) + ' nn_layers: ' + str(nn_layers) + ' \n' + pprint.pformat(results))

            if epoch % 10 == 0:
                print(best_results_info)
    return best_results


if __name__ == '__main__':
    # svm_classification()
    ffn_classification()