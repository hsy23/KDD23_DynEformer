import torch
import torch.nn as nn
from torch import optim

import pickle
import os
import time
import argparse
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset
import warnings
import matplotlib.pyplot as plt
import numpy as np

from Autoformer import Model as Autoformer
from Informer import Model as Informer

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

parser.add_argument('--freq', type=str, default='d',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
# forecasting task
parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
parser.add_argument('--label_len', type=int, default=12, help='start token length')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='1', help='device ids of multile gpus')

# save and test
parser.add_argument('--save_model', type=bool, default=True, help='save best model')


def get_mape(yTrue, yPred, scaler=None):
    if scaler:
        yTrue = scaler.inverse_transform(yTrue)
        yPred = scaler.inverse_transform(yPred)

    return np.mean(np.abs((yTrue - yPred) / yTrue) * 100)


def get_mse(yTrue, yPred, scaler=None):
    if scaler:
        yTrue = scaler.inverse_transform(yTrue)
        yPred = scaler.inverse_transform(yPred)

    return np.mean((yTrue - yPred) ** 2)


def get_mae(yTrue, yPred, scaler):
    if scaler:
        yTrue = scaler.inverse_transform(yTrue)
        yPred = scaler.inverse_transform(yPred)

    return np.mean(np.abs(yTrue - yPred))


class MinMaxScaler:
    def fit_transform(self, y):
        self.max = np.max(y)
        self.min = np.min(y)
        return (y - self.min) / (self.max - self.min)

    def inverse_transform(self, y):
        return y * (self.max - self.min) + self.min

    def transform(self, y):
        return (y - self.min) / (self.max - self.min)


class PPIO_Dataset(Dataset):
    def __init__(self, X, enc_len=48, label_len=12, pred_len=24):
        num_ts, num_periods, num_features = X.shape
        X_train_all = []
        Y_train_all = []
        X_mark_all = []
        Y_mark_all = []

        for i in range(num_ts):
            for j in range(enc_len, num_periods - pred_len, 12):
                X_train_all.append(X[i, j - enc_len:j, 0])
                Y_train_all.append(X[i, j - label_len:j + pred_len, 0])
                X_mark_all.append(X[i, j - enc_len:j, 1:4])
                Y_mark_all.append(X[i, j - label_len:j + pred_len, 1:4])

        self.X = np.asarray(X_train_all).reshape(-1, enc_len, 1)
        self.Y = np.asarray(Y_train_all).reshape(-1, label_len + pred_len, 1)
        self.X_mark = np.asarray(X_mark_all).reshape(-1, enc_len, 3)
        self.Y_mark = np.asarray(Y_mark_all).reshape(-1, label_len + pred_len, 3)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.X_mark[index], self.Y_mark[index]


def batch_generator_padding(X, enc_len=48, label_len=12, pred_len=24, step=2):
    '''
        Args:
        X (array like): shape (num_samples, num_features, num_periods)
        y (array like): shape (num_samples, num_periods)
        num_obs_to_train (int):
        seq_len (int): sequence/encoder/decoder length
        batch_size (int)
        '''
    num_ts, num_periods, n_feats = X.shape
    X_train_all = []
    Y_train_all = []
    X_mark_all = []
    Y_mark_all = []

    for i in range(num_ts):
        for j in range(enc_len, num_periods - pred_len, step):
            X_train_all.append(X[i, j - enc_len:j, 0])
            Y_train_all.append(X[i, j - label_len:j + pred_len, 0])
            X_mark_all.append(X[i, j - enc_len:j, 1:4])
            Y_mark_all.append(X[i, j - label_len:j + pred_len, 1:4])

    X = np.asarray(X_train_all).reshape(-1, enc_len, 1)
    Y = np.asarray(Y_train_all).reshape(-1, label_len + pred_len, 1)
    X_mark = np.asarray(X_mark_all).reshape(-1, enc_len, 3)
    Y_mark = np.asarray(Y_mark_all).reshape(-1, label_len + pred_len, 3)
    return X, Y, X_mark, Y_mark


def get_test_data(step):
    X = pickle.load(open(r"../../data/ECW_switch.pkl", 'rb'))
    y = X[:, :, 0]

    num_ts, num_periods, num_features = X.shape

    xscaler, yscaler = pickle.load(open('saved_model/dyne_data_scaler_tmp.pkl', 'rb'))

    Xte = xscaler.transform(X.reshape(-1, num_features)).reshape(num_ts, -1, num_features)

    X, Y, X_mark, Y_mark = batch_generator_padding(Xte, step=step)

    return X, Y, X_mark, Y_mark, yscaler


device = torch.device('cuda:0')
args = parser.parse_args()
used_model = 'Autoformer'

if used_model == 'Informer':
    model = torch.load('saved_model/Informer_best.pt')
else:
    model = torch.load('saved_model/Autoformer_best.pt')

# model = nn.DataParallel(model)
model = model.to(device)

criterion = nn.MSELoss().to(device)

X, Y, X_mark, Y_mark, yscaler = get_test_data(step=2)
X = torch.from_numpy(X.astype(float))
Y = torch.from_numpy(Y.astype(float))
X_mark = torch.from_numpy(X_mark.astype(float))
Y_mark = torch.from_numpy(Y_mark.astype(float))

with torch.no_grad():
    batch_x = X.float().to(device)
    batch_y = Y.float().to(device)
    batch_x_mark = X_mark.float().to(device)
    batch_y_mark = Y_mark.float().to(device)

    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, None)

    f_dim = -1
    outputs = outputs[:, -args.pred_len:, f_dim]
    batch_y = batch_y[:, -args.pred_len:, f_dim]

    test_loss = criterion(outputs, batch_y)
    test_mse = get_mse(outputs.cpu(), batch_y.cpu(), yscaler)
    test_mape = get_mape(outputs.cpu(), batch_y.cpu(), yscaler)
    test_mae = get_mae(outputs.cpu(), batch_y.cpu(), yscaler)

    print('The Mean Squared Error of forecasts is {}'.format(test_loss))
    print('The Mean Squared Error of forecasts is {} (raw)'.format(test_mse))
    print('The MAPE of forecasts is {} (raw)'.format(test_mape))
    print('The Mean Absolute Error of forecasts is {} (raw)'.format(test_mae))