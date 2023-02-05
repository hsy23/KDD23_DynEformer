
'''
Pytorch Implementation of MQ-RNN
Paper Link: https://arxiv.org/abs/1711.11053
Author: Jing Wang (jingw2@foxmail.com)
'''

import torch
from torch import nn

from torch.optim import Adam
import time
from MqRnn_model import MQRNN
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import util
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import argparse
from datetime import date
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.append('../')
from data_process_utils import *
from global_utils import *


class PPIO_Dataset(Dataset):
    def __init__(self, X, Y, enc_len=48, label_len=12, pred_len=24, num_static=12):
        self.pred_len = pred_len
        num_ts, num_periods, num_features = X.shape
        sq_len = enc_len + pred_len
        X_train_all = []
        Y_train_all = []
        X.astype(float)

        for i in range(num_ts):
            for j in range(sq_len, num_periods, 12):
                X_train_all.append(X[i, j - sq_len:j, :])
                Y_train_all.append(Y[i, j - sq_len:j])

        self.X = np.stack(X_train_all).reshape(-1, sq_len, num_features).astype(float)
        self.Y = np.stack(Y_train_all).reshape(-1, sq_len).astype(float)
        # self.Y = np.asarray(X[:,:,0]).reshape(-1, sq_len)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index, :48, :], self.Y[index, :48], self.X[index, -self.pred_len:, :], self.Y[index, -self.pred_len:]


def train(X, y, args, quantiles):
    num_ts, num_periods, num_features = X.shape
    num_quantiles = len(quantiles)
    device = torch.device('cuda:0')
    model = MQRNN(
        args.seq_len,
        num_quantiles,
        num_features,
        args.embedding_size,
        args.encoder_hidden_size,
        args.n_layers,
        args.decoder_hidden_size
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    Xtr, ytr, Xte, yte = train_test_split(X, y)
    losses = []
    test_losses = []
    mse_loss = nn.MSELoss().to(device)

    yscaler = MinMaxScaler()
    ytr = yscaler.fit_transform(ytr)
    # yte = yscaler.fit_transform(yte)

    # pickle.dump(yscaler, open('8_scalers_new.pkl', 'wb'))

    num_obs_to_train = args.num_obs_to_train
    seq_len = args.seq_len
    Xtr_loader = DataLoader(PPIO_Dataset(Xtr, ytr), batch_size=args.batch_size)
    Xte_loader = DataLoader(PPIO_Dataset(Xte, yte), batch_size=args.batch_size)

    min_loss = 1000
    for epoch in tqdm(range(args.num_epoches)):
        # print("Epoch {} start...".format(epoch))
        train_epoch_loss = []
        for X_train_batch, y_train_batch, Xf, yf in Xtr_loader:
            X_train_tensor = X_train_batch.float().to(device)
            y_train_tensor = y_train_batch.float().to(device)
            Xf = Xf.float().to(device)
            yf = yf.float().to(device)

            ypred = model(X_train_tensor, y_train_tensor, Xf)

            # quantile loss
            loss = torch.zeros_like(yf)
            num_ts = Xf.size(0)
            for q, rho in enumerate(quantiles):
                ypred_rho = ypred[:, :, q].view(num_ts, -1)
                e = ypred_rho - yf
                loss += torch.max(rho * e, (rho - 1) * e)
            loss = loss.mean()
            mse_l = mse_loss(ypred[:, :, 1], yf)
            train_epoch_loss.append(mse_l.item())
            losses.append(mse_l.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(np.average(train_epoch_loss))

        # test
        with torch.no_grad():
            test_epoch_loss = []
            test_epoch_mse = []
            test_epoch_mae = []

            for Xtest, ytest, Xf_t, yf_t in Xte_loader:
                if yscaler is not None:
                    ytest = yscaler.transform(ytest)

                Xtest_tensor = Xtest.float().to(device)
                ytest_tensor = ytest.float().to(device)
                Xf_t = Xf_t.float().to(device)
                yf_t_tensor = yf_t.float().to(device)

                ypred = model(Xtest_tensor, ytest_tensor, Xf_t)
                ypred = ypred[:, :, 1]

                test_epoch_loss.append(mse_loss(ypred, yscaler.transform(yf_t_tensor)).item())
                ypred = ypred.cpu().numpy()

                if yscaler is not None:
                    ypred = yscaler.inverse_transform(ypred)
                test_epoch_mse.append(((ypred.reshape(-1) - np.asarray(yf_t).reshape(-1)) ** 2).mean())
                test_epoch_mae.append(np.abs(ypred.reshape(-1) - np.asarray(yf_t).reshape(-1)).mean())

            test_losses.append(np.average(test_epoch_loss))
            print('loss:{}, test Loss:{}, mse:{}, mae:{}'.format(losses[-1], test_losses[-1],
                                                                 np.average(test_epoch_mse), np.average(test_epoch_mae)))
        if args.save_model:
            if test_losses[-1] < min_loss:
                best_model = model
                min_loss = test_losses[-1]
                torch.save(model, 'MQRNN_ppio_best.pt')

    return losses, test_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoches", "-e", type=int, default=300)
    parser.add_argument("--step_per_epoch", "-spe", type=int, default=2)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("--n_layers", "-nl", type=int, default=3)
    parser.add_argument("--encoder_hidden_size", "-ehs", type=int, default=64)
    parser.add_argument("--decoder_hidden_size", "-dhs", type=int, default=64)
    parser.add_argument("--seq_len", "-sl", type=int, default=24)
    parser.add_argument("--num_obs_to_train", "-not", type=int, default=24*2)
    parser.add_argument("--embedding_size", "-es", type=int, default=10)
    parser.add_argument("--standard_scaler", "-ss", action="store_true")
    parser.add_argument("--log_scaler", "-ls", action="store_true")
    parser.add_argument("--mean_scaler", "-ms", action="store_true")
    parser.add_argument("--minmax_scaler", "-mm", action="store_true", default=True)
    parser.add_argument("--show_plot", "-sp", action="store_true", default=True)
    parser.add_argument("--run_test", "-rt", action="store_true", default=True)
    parser.add_argument("--batch_size", "-b", type=int, default=256)
    parser.add_argument("--save_model", type=bool, default=False)
    args = parser.parse_args()

    if args.run_test:
        X_all = np.load(open(r"../../../raw_data/X_all_0801_0830.npy", 'rb'), allow_pickle=True)
        y_all = np.load(open(r"../../../raw_data/y_all_0801_0830.npy", 'rb'), allow_pickle=True)
        X_all = X_all[:, :, 1:4]
        quantiles = [0.1, 0.5, 0.9]
        losses, test_losses = train(X_all, y_all, args, quantiles)

        # if args.show_plot:
        #     plt.plot(range(len(losses)), losses, "k-")
        #     plt.xlabel("Period")
        #     plt.ylabel("Loss")
        #     plt.show()
