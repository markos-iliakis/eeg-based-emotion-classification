import logging
import os
import pprint
from statistics import mean

import pandas
import torch.nn
from matplotlib import pyplot as plt
from torch import nn, IntTensor
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
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
    features = ['de']  # 'de', 'psd', 'dasm', 'rasm', 'asm', 'dcau'
    smoothing = ['movingAve']  # 'movingAve', 'LDS'
    features = [f + '_' + s for f in features for s in smoothing]

    # Available bands: delta (1-3Hz)/ theta (4-7Hz)/ alpha (8-13Hz)/ beta (14-30Hz)/ gamma(31-50Hz)
    bands = {'delta': 0,
             'theta': 1,
             'alpha': 2,
             'beta': 3,
             'gamma': 4}

    # Available kernels:
    kernels = ['linear']  # , 'gak', 'linear', 'sigmoid'

    best_results = {}
    best_results_info = ''
    for band in bands.keys():

        band = 'beta'
        for feature in features:

            train_x, train_y = get_data(starting_per=1, ending_per=14, feature=feature, band=bands[band])
            validation_x, validation_y = get_data(starting_per=14, ending_per=16, feature=feature, band=bands[band])

            for kernel in kernels:

                clf = svm.TimeSeriesSVC(kernel=kernel)
                clf.fit(train_x, train_y)
                pred_y = clf.predict(validation_x)

                results = metrics.classification_report(validation_y, pred_y, output_dict=True)
                df = pandas.DataFrame(results).transpose()
                df.to_excel('svm_logs/best_svm_classification_report.xlsx')
                if not best_results:
                    best_results = results

                if results['macro avg']['f1-score'] >= best_results['macro avg']['f1-score']:
                    best_results = results
                    best_results_info = 'feature: ' + feature + ' band: ' + band + ' kernel: ' + kernel + ' \n' + pprint.pformat(
                        results)

                print('feature: ' + feature + ' band: ' + band + ' kernel: ' + kernel + ' \n' + pprint.pformat(results))
        break

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

    def forward(self, x):
        # Linear layer
        output = self.linear(x)

        if not self.last_layer:
            # non-linearity
            output = self.act(output)
            # dropout
            output = self.dropout(output)
            # skip connection
            # if not self.first_layer:
            #     output = output + input
            # layer normalization
            # output = self.layernorm(output)
        else:
            output = self.softmax(output)

        return output


class MultiLayerFeedForwardNN(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_layers=0, dropout_rate=None, hidden_dim=-1, device='cpu'):
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

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)

        return x


class EEGDataset(Dataset):
    def __init__(self, x, y):
        self.X = np.array(x)
        self.Y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index].astype(np.float32)), torch.from_numpy(np.array(self.Y[index]))


def ffn_classification():
    epochs = 100
    batch_size = 32
    nn_layers = 2
    dropout = 0
    hidden_dim = 4
    lr = 0.01
    device = 'cuda'
    steps = 5

    # keep logs
    ffn_logs_path = './ffn_logs/'
    if not os.path.exists(ffn_logs_path):
        os.mkdir(ffn_logs_path)

    # Available features: de / psd / dasm / rasm / asm / dcau | Available smoothing: movingAve / LDS
    features = ['de']  # 'de', 'psd', 'dasm', 'rasm', 'asm', 'dcau'
    smoothing = ['movingAve']  # 'movingAve', 'LDS'
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
        train_x, train_y = get_data(starting_per=1, ending_per=13, feature=feature, band=bands[band])
        validation_x, validation_y = get_data(starting_per=13, ending_per=15, feature=feature, band=bands[band])
        test_x, test_y = get_data(starting_per=15, ending_per=16, feature=feature, band=bands[band])

        channels = len(train_x[0])
        sec = len(train_x[0][0])
        train_dataset = EEGDataset(train_x, train_y)
        validation_dataset = EEGDataset(validation_x, validation_y)
        test_dataset = EEGDataset(test_x, test_y)

        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        ffn = MultiLayerFeedForwardNN(input_dim=channels*2, output_dim=3, num_hidden_layers=nn_layers, dropout_rate=dropout,
                                      hidden_dim=hidden_dim, device=device)
        ffn.to(device)

        criterion = CrossEntropyLoss()

        # optimizer = optim.Adam(ffn.parameters(), lr=lr)
        optimizer = SGD(ffn.parameters(), lr=lr)

        mean_losses = []
        mean_validation_losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()

            losses = []
            ffn.train()
            # Train
            for train_batch, train_batch_labels in train_loader:
                train_batch, train_batch_labels = train_batch.to(device), train_batch_labels.to(device)

                # train_batch = train_batch.view(-1, channels*sec).requires_grad_()
                # train_batch = torch.mean(train_batch, dim=2, keepdim=False)
                # train_batch = (torch.stack(torch.std_mean(train_batch, dim=1, keepdim=False), dim=2)).view(-1, sec*2)
                train_batch = (torch.stack(torch.std_mean(train_batch, dim=2, keepdim=False), dim=2)).view(-1, channels*2)
                train_batch.requires_grad_()
                output = ffn(train_batch)
                loss = criterion(output, train_batch_labels)
                losses.append(IntTensor.item(loss))
                loss.backward()
                optimizer.step()

            mean_losses.append(mean(losses))

            if (epoch+1) % steps == 0:
                # Calc metrics for validation
                validation_losses = []
                validation_pred = []
                validation_labels = []
                ffn.eval()
                with torch.no_grad():
                    for validation_batch, validation_batch_labels in validation_loader:
                        validation_batch, validation_batch_labels = validation_batch.to(device), validation_batch_labels.to(device)

                        # validation_batch = validation_batch.view(-1, channels*sec)
                        # validation_batch = torch.mean(validation_batch, dim=1, keepdim=False)
                        # validation_batch = (torch.stack(torch.std_mean(validation_batch, dim=1, keepdim=False), dim=2)).view(-1, sec*2)
                        validation_batch = (torch.stack(torch.std_mean(validation_batch, dim=2, keepdim=False), dim=2)).view(-1, channels*2)
                        output = ffn(validation_batch)
                        validation_loss = criterion(output, validation_batch_labels)
                        validation_losses.append(IntTensor.item(validation_loss))

                        _, predicted = torch.max(output, 1)

                        validation_labels.extend(validation_batch_labels.tolist())
                        validation_pred.extend(predicted.tolist())

                mean_validation_losses.append(mean(validation_losses))

                results = metrics.classification_report(validation_labels, validation_pred, output_dict=True, zero_division=0)

                if not best_results:
                    best_results = results

                if results['accuracy'] >= best_results['accuracy']:
                    torch.save(ffn, 'best-model.pt')
                    best_results = results
                    best_results_info = f'feature: {feature} band: {band} epoch: {str(epoch)} lr: {str(lr)} nn_layers: {str(nn_layers)}\n {pprint.pformat(results)}'

                print(f'Epoch {epoch} ........... Mean Train Loss: {mean(mean_losses)} Mean Test Loss: {mean(mean_validation_losses)}')
                print(pprint.pformat(results))

        # Test
        test_labels = []
        test_pred = []
        model = torch.load('best-model.pt')
        with torch.no_grad():
            for test_batch, test_batch_labels in test_loader:
                test_batch, test_batch_labels = test_batch.to(device), test_batch_labels.to(
                    device)

                # test_batch = test_batch.view(-1, channels*sec)
                # test_batch = torch.mean(test_batch, dim=1, keepdim=False)
                # test_batch = (torch.stack(torch.std_mean(test_batch, dim=1, keepdim=False), dim=2)).view(-1, sec * 2)
                test_batch = (torch.stack(torch.std_mean(test_batch, dim=2, keepdim=False), dim=2)).view(-1, channels * 2)
                output = model(test_batch)

                _, predicted = torch.max(output, 1)

                test_labels.extend(test_batch_labels.tolist())
                test_pred.extend(predicted.tolist())
        test_results = metrics.classification_report(test_labels, test_pred, output_dict=True, zero_division=0)
        print("Test results: ")
        pprint.pprint(test_results)

        plot_loss_epochs(mean_losses, mean_validation_losses, epochs-1, steps)
    print(best_results_info)
    return best_results


def plot_loss_epochs(train_loss, val_loss, epochs, steps):
    plt.plot(range(1, epochs + 2), train_loss, 'g', label='Training loss')
    plt.plot(range(1, epochs + 2, steps), val_loss, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # svm_classification()
    ffn_classification()
