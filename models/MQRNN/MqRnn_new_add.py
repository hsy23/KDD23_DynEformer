#!/usr/bin/python 3.6
# -*-coding:utf-8-*-

'''
Pytorch Implementation of MQ-RNN
Paper Link: https://arxiv.org/abs/1711.11053
Author: Jing Wang (jingw2@foxmail.com)
'''

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import time

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

import sys
sys.path.append('../')
from data_process_utils import *
from global_utils import *


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
        for j in range(num_obs_to_train, num_periods - seq_len, 2):
            X_train_all.append(X[i, j-num_obs_to_train:j, :])
            Y_train_all.append(Y[i, j-num_obs_to_train:j])
            Xf_all.append(X[i, j:j+seq_len])
            Yf_all.append(Y[i, j:j+seq_len])

    X_train_all = np.asarray(X_train_all).reshape(-1, num_obs_to_train, n_feats)
    Y_train_all = np.asarray(Y_train_all).reshape(-1, num_obs_to_train)
    Xf_all = np.asarray(Xf_all).reshape(-1, seq_len, n_feats)
    Yf_all = np.asarray(Yf_all).reshape(-1, seq_len)
    return X_train_all, Y_train_all, Xf_all, Yf_all


def inference(X, y, args):
    model = torch.load('MQRNN_ppio_best.pt')
    device = torch.device('cuda:0')

    num_ts, num_periods, num_features = X.shape
    model = model.to(device)

    random.seed(2)
    mse = nn.MSELoss().to(device)

    Xte = X
    yte = y

    yscaler = MinMaxScaler() #  pickle.load(open('8_scalers.pkl', 'rb'))
    yscaler.fit(yte)
    # yscaler = pickle.load(open('8_scalers_new.pkl', 'rb'))

    pre_seq_len = args.seq_len
    num_obs_to_train = args.num_obs_to_train
    X_test_all, Y_test_all, Xf_t_all, Yf_t_all = batch_generator_all(Xte, yte, num_obs_to_train, pre_seq_len)

    with torch.no_grad():
        test_epoch_loss = []
        test_epoch_mse = []
        test_epoch_mae = []

        Xtest, ytest = X_test_all, Y_test_all

        ytest = yscaler.transform(ytest)

        Xtest_tensor = torch.from_numpy(Xtest.astype(float)).float().to(device)
        ytest_tensor = torch.from_numpy(ytest.astype(float)).float().to(device)
        Xf_t = torch.from_numpy(Xf_t_all.astype(float)).float().to(device)
        yf_t_tensor = torch.from_numpy(Yf_t_all.astype(float)).float().to(device)
        ypred = model(Xtest_tensor, ytest_tensor, Xf_t)
        ypred = ypred[:, :, 1]

        test_epoch_loss.append(mse(ypred, yscaler.transform(yf_t_tensor)).item())
        ypred = ypred.cpu().numpy()

        ypred = yscaler.inverse_transform(ypred)

        test_epoch_mse.append(((ypred.reshape(-1) - Yf_t_all.reshape(-1)) ** 2).mean())
        test_epoch_mae.append(np.abs(ypred.reshape(-1) - Yf_t_all.reshape(-1)).mean())

        print('The Test MSE Loss is {}'.format(np.average(test_epoch_loss)))
        print('The Mean Squared Error of forecasts is {} (raw)'.format(np.average(test_epoch_mse)))
        print('The Mean Absolute Error of forecasts is {} (raw)'.format(np.average(test_epoch_mae)))


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
    parser.add_argument("--day_periods", "-per", type=int, default=288)
    parser.add_argument("--num_days", "-ds", type=int, default=7)
    args = parser.parse_args()

    if args.run_test:
        X_all = pickle.load(open(r"../../data/ECW_newapp.pkl", 'rb'))
        y_all = X_all[:, :, 0]
        X_all = X_all[:, :, 1:4]
        losses, test_losses = inference(X_all, y_all, args)