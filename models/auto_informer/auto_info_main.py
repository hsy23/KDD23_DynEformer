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

from models.global_utils import train_test_split

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
# forecasting task
parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
parser.add_argument('--label_len', type=int, default=12, help='start token length')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

# model define
parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder auto_info_layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder auto_info_layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

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
                X_train_all.append(X[i, j-enc_len:j, 0])
                Y_train_all.append(X[i, j-label_len:j+pred_len, 0])
                X_mark_all.append(X[i, j-enc_len:j, 1:4])
                Y_mark_all.append(X[i, j-label_len:j+pred_len, 1:4])

        self.X = np.asarray(X_train_all).reshape(-1, enc_len, 1)
        self.Y = np.asarray(Y_train_all).reshape(-1, label_len+pred_len, 1)
        self.X_mark = np.asarray(X_mark_all).reshape(-1, enc_len, 3)
        self.Y_mark = np.asarray(Y_mark_all).reshape(-1, label_len+pred_len, 3)
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.X_mark[index], self.Y_mark[index]


def get_ppio(batch_size=256):
    X = np.load(open(r"../../data/ECW_08.npy", 'rb'), allow_pickle=True)
    y = X[:, :, 0]

    Xtr, ytr, Xte, yte = train_test_split(X, y)
    num_ts, num_periods, num_features = Xte.shape

    xscaler = preprocessing.MinMaxScaler()
    yscaler = preprocessing.MinMaxScaler()
    yscaler.fit(Xtr[:,:,0].reshape(-1, 1))
    Xtr = xscaler.fit_transform(Xtr.reshape(-1, num_features)).reshape(num_ts, -1, num_features)
    Xte = xscaler.transform(Xte.reshape(-1, num_features)).reshape(num_ts, -1, num_features)

    # pickle.dump([xscaler, yscaler], open('8_scalers.pkl', 'wb'))

    Xte_loader = DataLoader(PPIO_Dataset(Xte), batch_size = batch_size)
    Xtr_loader = DataLoader(PPIO_Dataset(Xtr), batch_size = batch_size)
    
    return Xtr_loader, Xte_loader, yscaler

device = torch.device('cuda:0')
args = parser.parse_args()
epochs = 300
used_model = 'Autoformer'

if used_model == 'Informer':
    model = Informer(args).float()
else:
    model = Autoformer(args).float()

# model = nn.DataParallel(model)
model = model.to(device)

criterion = nn.MSELoss()
model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
Xtr_loader, Xte_loader, yscaler = get_ppio()
min_loss = 1000
best_model = None

train_loss = []
test_loss = []
test_mape = []
test_mse = []
test_mae = []
for epoch in range(epochs):
    epo_train_losses = []
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(Xtr_loader):
        model_optim.zero_grad()
        
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, None)

        f_dim = -1
        outputs = outputs[:, -args.pred_len:, f_dim:]
        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
        loss = criterion(outputs, batch_y)
        
        epo_train_losses.append(loss.item())

        loss.backward()
        model_optim.step()
    train_loss.append(np.mean(epo_train_losses))

    epo_test_losses = []
    epo_mse = []
    epo_mape = []
    epo_mae = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(Xte_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, None)

            f_dim = -1
            outputs = outputs[:, -args.pred_len:, f_dim]
            batch_y = batch_y[:, -args.pred_len:, f_dim]
            loss = criterion(outputs, batch_y)
            epo_test_losses.append(loss.item())
            epo_mse.append(get_mse(outputs.cpu(), batch_y.cpu(), yscaler))
            epo_mape.append(get_mape(outputs.cpu(), batch_y.cpu(), yscaler))
            epo_mae.append(get_mae(outputs.cpu(), batch_y.cpu(), yscaler))
    
    test_loss.append(np.mean(epo_test_losses))
    test_mse.append(np.mean(epo_mse))
    test_mape.append(np.mean(epo_mape))
    test_mae.append(np.mean(epo_mae))

    if args.save_model:
        if test_loss[-1] < min_loss:
            best_model = model
            min_loss = test_loss[-1]
            torch.save(model, 'saved_model/{}_best.pt'.format(used_model))

    print(f'epoch {epoch}, train loss: {train_loss[-1]}, test loss: {test_loss[-1]}, mse: {test_mse[-1]}, mape: {test_mape[-1]}, mae: {test_mae[-1]}')

print(np.min(test_mse), np.min(test_mape), np.min(test_mae))