import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
import argparse
import math
from deep_trans_model import Transformer
from sklearn import preprocessing
from tqdm import tqdm
from torch.optim import Adam
import torch
import time

import sys

from torch.utils.data import DataLoader, Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--standard_scaler", "-ss", action="store_true")
parser.add_argument("--log_scaler", "-ls", action="store_true")
parser.add_argument("--mean_scaler", "-ms", action="store_true")
parser.add_argument("--minmax_scaler", "-mm", action="store_true", default=True)

parser.add_argument("--num_epoches", "-e", type=int, default=300)
parser.add_argument("--step_per_epoch", "-spe", type=int, default=3)
parser.add_argument("-lr", type=float, default=1e-4)
parser.add_argument("--batch_size", "-b", type=int, default=256)

parser.add_argument("--n_encoder_layers", "-nel", type=int, default=2)
parser.add_argument("--n_decoder_layers", "-ndl", type=int, default=1)
parser.add_argument("--d_model", "-dm", type=int, default=512)  # 嵌入维度
parser.add_argument("--nhead", "-nh", type=int, default=8)  # 注意力头数量
parser.add_argument("--dim_feedforward", "-hs", type=int, default=512)
parser.add_argument("--dec_seq_len", "-dl", type=int, default=12)  # decoder用到的输入长度
parser.add_argument("--out_seq_len", "-ol", type=int, default=24)  # 预测长度
parser.add_argument("--enc_seq_len", "-not", type=int, default=24*2)  # 输入训练长度
parser.add_argument("-dropout", type=float, default=0.05)
parser.add_argument("-activation", type=str, default='relu')

parser.add_argument("--run_test", "-rt", action="store_true", default=True)
parser.add_argument("--save_model", "-sm", type=bool, default=True)
parser.add_argument("--load_model", "-lm", type=bool, default=False)
parser.add_argument("--show_plot", "-sp", type=bool, default=True)

parser.add_argument("--day_periods", "-dp", type=int, default=288)
parser.add_argument("--num_periods", "-np", type=int, default=24)
parser.add_argument("--num_days", "-ds", type=int, default=30)

args = parser.parse_args()


class Cloud_Dataset(Dataset):
    def __init__(self, X, enc_len=48, label_len=12, pred_len=24):
        self.pred_len = pred_len
        self.enc_len = enc_len
        num_ts, num_periods, num_features = X.shape
        sq_len = enc_len+pred_len
        X_train_all = []
        # X_static = []
        X.astype(float)

        for i in range(num_ts):
            for j in range(sq_len, num_periods, 12):
                X_train_all.append(X[i, j-sq_len:j, :])
                # X_static.append(X[i, j-sq_len:j, :])

        self.X = np.stack(X_train_all).reshape(-1, sq_len, num_features)
        # self.X_static = np.stack(X_static).reshape(-1, sq_len, )
        # self.Y = np.asarray(X[:,:,0]).reshape(-1, sq_len)
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index, :self.enc_len, 0:4], self.X[index, -self.pred_len:, 0]


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

device = torch.device('cuda:0')

model = Transformer(args.d_model, args.d_model, 4, args.dec_seq_len, args.out_seq_len,
                        n_encoder_layers=args.n_encoder_layers, n_decoder_layers=args.n_decoder_layers,
                        n_heads=args.nhead, dropout=args.dropout)
model = model.to(device)
optimizer = Adam(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()
batch_size = 32
epochs = 300

Xtr = np.load('../../../raw_data/tr_vm.npy')
Xte = np.load('../../../raw_data/te_vm.npy')
Xtr[:,:,0], Xte[:,:,0] = Xtr[:,:,0]/10, Xte[:,:,0]/10
ytr = Xtr[:,:,0]
yte = Xtr[:,:,0]
xscaler = preprocessing.MinMaxScaler()
yscaler = preprocessing.MinMaxScaler()
num_ts, num_periods, num_features = Xtr.shape
Xtr = xscaler.fit_transform(Xtr.reshape(-1, num_features)).reshape(num_ts, num_periods, num_features)
num_ts, num_periods, num_features = Xte.shape
Xte = xscaler.transform(Xte.reshape(-1, num_features)).reshape(num_ts, num_periods, num_features)
yscaler.fit(ytr.reshape(-1, 1))
Xtr_loader = DataLoader(Cloud_Dataset(Xtr), batch_size = batch_size, shuffle=True)
Xte_loader = DataLoader(Cloud_Dataset(Xte), batch_size=batch_size, shuffle=False)

train_loss = []
test_loss = []
test_mape = []
test_mse = []
test_mae = []
min_loss=1000
for epoch in range(epochs):
    epo_train_losses = []
    for x, y in Xtr_loader:
        optimizer.zero_grad()
        x = x.float().to(device)
        y = y.float().to(device)

        yPred = model(x)
        loss = criterion(yPred, y)
        epo_train_losses.append(loss.item())

        loss.backward()
        optimizer.step()
    train_loss.append(np.mean(epo_train_losses))

    epo_test_losses = []
    epo_mse = []
    epo_mape = []
    epo_mae = []
    model.eval()
    with torch.no_grad():
        for x, y in Xte_loader:
            x = x.float().to(device)
            # xst = xst.float().to(device)
            y = y.float().to(device)

            yPred = model(x)

            loss = criterion(yPred, y)

            epo_test_losses.append(loss.item())
            epo_mse.append(get_mse(yPred.cpu(), y.cpu(), yscaler))
            epo_mape.append(get_mape(yPred.cpu(), y.cpu(), yscaler))
            epo_mae.append(get_mae(yPred.cpu(), y.cpu(), yscaler))
    
    test_loss.append(np.mean(epo_test_losses))
    test_mse.append(np.mean(epo_mse))
    test_mape.append(np.mean(epo_mape))
    test_mae.append(np.mean(epo_mae))


    print(f'epoch {epoch}, train loss: {train_loss[-1]}, test loss: {test_loss[-1]}, mse: {test_mse[-1]}, mape: {test_mape[-1]}, mae: {test_mae[-1]}')
print(np.min(test_mse), np.min(test_mape), np.min(test_mae))