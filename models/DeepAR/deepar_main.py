#!/usr/bin/python 3.7
# -*-coding:utf-8-*-

'''
DeepAR Model (Pytorch Implementation)
Paper Link: https://arxiv.org/abs/1704.04110
Author: Jing Wang (jingw2@foxmail.com)
'''

import torch
from torch import nn
from torch.optim import Adam
import time

import numpy as np
import os
import random
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
import util
from datetime import date
import argparse
from tqdm import tqdm
from deepar_model import DeepAR



def batch_generator_all(X, Y, num_obs_to_train, seq_len):
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
    Xf_all = []
    Yf_all = []

    for i in range(num_ts):
        for j in range(num_obs_to_train, num_periods - seq_len, 12):
            X_train_all.append(X[i, j-num_obs_to_train:j, :])
            Y_train_all.append(Y[i, j-num_obs_to_train:j])
            Xf_all.append(X[i, j:j+seq_len])
            Yf_all.append(Y[i, j:j+seq_len])

    X_train_all = np.asarray(X_train_all).reshape(-1, num_obs_to_train, n_feats).astype(float)
    Y_train_all = np.asarray(Y_train_all).reshape(-1, num_obs_to_train).astype(float)
    Xf_all = np.asarray(Xf_all).reshape(-1, seq_len, n_feats).astype(float)
    Yf_all = np.asarray(Yf_all).reshape(-1, seq_len).astype(float)
    return X_train_all, Y_train_all, Xf_all, Yf_all


def train(X, y, args):
    '''
    Args:
    - X (array like): shape (num_samples, num_features, num_periods)
    - y (array like): shape (num_samples, num_periods)
    - epoches (int): number of epoches to run
    - step_per_epoch (int): steps per epoch to run
    - seq_len (int): output horizon
    - likelihood (str): what type of likelihood to use, default is gaussian
    - num_skus_to_show (int): how many skus to show in test phase
    - num_results_to_sample (int): how many samples in test phase as prediction
    '''

    device = torch.device('cuda:0')

    num_ts, num_periods, num_features = X.shape
    model = DeepAR(num_features, args.embedding_size,
                   args.hidden_size, args.n_layers, args.lr, args.likelihood)
    # model = pickle.load(open("deepar.pkl", 'rb'))
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    random.seed(2)
    # select sku with most top n quantities
    Xtr, ytr, Xte, yte = train_test_split(X, y)

    if args.likelihood == "g":
        loss_func = util.gaussian_likelihood_loss
    elif args.likelihood == "nb":
        loss_func = util.negative_binomial_loss

    losses = []
    test_losses = []
    mse_loss = nn.MSELoss().to(device)
    cnt = 0

    yscaler = None
    if args.standard_scaler:
        yscaler = util.StandardScaler()
    elif args.log_scaler:
        yscaler = util.LogScaler()
    elif args.mean_scaler:
        yscaler = util.MeanScaler()
    elif args.minmax_scaler:
        yscaler = MinMaxScaler()
    if yscaler is not None:
        ytr = yscaler.fit_transform(ytr)

    # training
    seq_len = args.seq_len
    num_obs_to_train = args.num_obs_to_train
    X_train_all, Y_train_all, Xf_all, Yf_all = batch_generator_all(Xtr, ytr, num_obs_to_train, seq_len)
    X_test_all, Y_test_all, Xf_t_all, Yf_t_all = batch_generator_all(Xte, yte, num_obs_to_train, seq_len)
    for epoch in tqdm(range(args.num_epoches)):
        # print("Epoch {} starts...".format(epoch))
        train_epoch_loss = []
        for step in range(int(len(X_train_all)/args.batch_size)):
            be = step*args.batch_size
            end = (step+1)*args.batch_size
            Xtrain, ytrain, Xf, yf = X_train_all[be:end, :, :], Y_train_all[be:end, :], \
                                     Xf_all[be:end, :, :], Yf_all[be:end, :]
            Xtrain_tensor = torch.from_numpy(Xtrain).float().to(device)
            ytrain_tensor = torch.from_numpy(ytrain).float().to(device)
            Xf = torch.from_numpy(Xf).float().to(device)
            yf = torch.from_numpy(yf).float().to(device)
            ypred, mu, sigma = model(Xtrain_tensor, ytrain_tensor, Xf)
            # ypred_rho = ypred
            # e = ypred_rho - yf
            # loss = torch.max(rho * e, (rho - 1) * e).mean()
            ## gaussian loss
            ytrain_tensor = torch.cat([ytrain_tensor, yf], dim=1)
            loss = loss_func(ytrain_tensor, mu, sigma)
            mse_l = mse_loss(ypred, yf)
            train_epoch_loss.append(mse_l.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1

        losses.append(np.average(train_epoch_loss))
        print('The MSE Loss {}'.format(losses[-1]))


        # test
        with torch.no_grad():
            test_epoch_loss = []
            test_epoch_mse = []
            test_epoch_mae = []

            for step in range(int(len(X_test_all)/args.batch_size)):
                be = step*args.batch_size
                end = (step+1)*args.batch_size
                Xtest, ytest, Xf_t, yf_t = X_test_all[be:end, :, :], Y_test_all[be:end, :], \
                                         Xf_t_all[be:end, :, :], Yf_t_all[be:end, :]

                if yscaler is not None:
                    ytest = yscaler.transform(ytest)

                Xtest_tensor = torch.from_numpy(Xtest).float().to(device)
                ytest_tensor = torch.from_numpy(ytest).float().to(device)
                Xf_t = torch.from_numpy(Xf_t).float().to(device)
                yf_t_tensor = torch.from_numpy(yf_t).float().to(device)
                ypred, mu, sigma = model(Xtest_tensor, ytest_tensor, Xf_t)

                test_epoch_loss.append(mse_loss(ypred, yscaler.transform(yf_t_tensor)).item())
                ypred = ypred.cpu().numpy()

                if yscaler is not None:
                    ypred = yscaler.inverse_transform(ypred)
                test_epoch_mse.append(((ypred.reshape(-1) - yf_t.reshape(-1)) ** 2).mean())
                test_epoch_mae.append(np.abs(ypred.reshape(-1) - yf_t.reshape(-1)).mean())

            test_losses.append(np.average(test_epoch_loss))
            if epoch % 10 == 0:
                print('The Test MSE Loss is {}'.format(np.average(test_epoch_loss)))
                print('The Mean Squared Error of forecasts is {} (raw)'.format(np.average(test_epoch_mse)))
                print('The Mean Absolute Error of forecasts is {} (raw)'.format(np.average(test_epoch_mae)))
    return losses, test_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoches", "-e", type=int, default=500)
    parser.add_argument("--step_per_epoch", "-spe", type=int, default=300)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("--n_layers", "-nl", type=int, default=3)
    parser.add_argument("--hidden_size", "-hs", type=int, default=64)
    parser.add_argument("--embedding_size", "-es", type=int, default=64)
    parser.add_argument("--likelihood", "-l", type=str, default="g")
    parser.add_argument("--seq_len", "-sl", type=int, default=24)
    parser.add_argument("--num_obs_to_train", "-not", type=int, default=24*2)
    parser.add_argument("--num_results_to_sample", "-nrs", type=int, default=10)
    parser.add_argument("--show_plot", "-sp", action="store_true", default=True)
    parser.add_argument("--run_test", "-rt", action="store_true", default=True)
    parser.add_argument("--standard_scaler", "-ss", action="store_true")
    parser.add_argument("--log_scaler", "-ls", action="store_true")
    parser.add_argument("--mean_scaler", "-ms", action="store_true")
    parser.add_argument("--minmax_scaler", "-mm", action="store_true", default=True)

    parser.add_argument("--batch_size", "-b", type=int, default=256)
    parser.add_argument("--save_model", type=bool, default=False)

    parser.add_argument("--sample_size", type=int, default=10)  # 为每个预测生成的候选数量
    parser.add_argument("--day_periods", "-per", type=int, default=288)
    parser.add_argument("--num_days", "-ds", type=int, default=30)

    args = parser.parse_args()

    if args.run_test:
        X_all = np.load(open(r"../../data/ECW_08.npy", 'rb'), allow_pickle=True)
        y_all = X_all[:, :, 0]  # the target workload
        X_all = X_all[:, :, 1:4]
        losses, test_losses = train(X_all, y_all, args)
