import pickle

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
sys.path.append('../')
sys.path.append('../GlobalPooing')  # 添加路径以找到类

from data_process_utils import *
from global_utils import *

from torch.utils.data import DataLoader


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

    for i in range(num_ts):
        for j in range(num_obs_to_train, num_periods - seq_len, 12):
            X_train_all.append(X[i, j-num_obs_to_train:j, :])
            Y_train_all.append(Y[i, j:j+seq_len])

    X_train_all = np.asarray(X_train_all).reshape(-1, num_obs_to_train, n_feats)
    Y_train_all = np.asarray(Y_train_all).reshape(-1, seq_len)
    return X_train_all, Y_train_all


def reference(X, y, args):
    model = torch.load('DeepTrans_best.pt')
    device = torch.device('cuda:0')

    num_ts, num_periods, num_features = X.shape
    model = model.to(device)

    random.seed(2)
    mse = nn.MSELoss().to(device)

    Xte = X
    yte = y

    xscaler, yscaler = pickle.load(open('8_scalers.pkl', 'rb'))
    Xte = xscaler.transform(Xte.reshape(-1, num_features)).reshape(num_ts, -1, num_features)

    pre_seq_len = args.out_seq_len
    num_obs_to_train = args.enc_seq_len
    X_test_all, Y_test_all = batch_generator_all(Xte, yte, num_obs_to_train, pre_seq_len)

    with torch.no_grad():
        test_epoch_loss = []
        test_epoch_mse = []
        test_epoch_mae = []

        Xtest, ytest = X_test_all, Y_test_all

        Xtest_tensor = torch.from_numpy(Xtest.astype(float)).float().to(device)
        ytest_tensor = torch.from_numpy(ytest.astype(float)).float().to(device)

        yPred_test = model(Xtest_tensor)

        test_epoch_loss.append(mse(yPred_test, yscaler.transform(ytest_tensor)).item())
        yPred_test = yPred_test.cpu().numpy()

        if yscaler is not None:
            yPred_test = yscaler.inverse_transform(yPred_test)

        test_epoch_mse.append(((ytest.reshape(-1) - yPred_test.reshape(-1)) ** 2).mean())
        test_epoch_mae.append(np.abs(ytest.reshape(-1) - yPred_test.reshape(-1)).mean())

        pickle.dump([yPred_test, ytest], open('../../draw_pics/draw_new/dptrans_newmac_plot.pkl', 'wb'))

        print('The Test MSE Loss is {}'.format(np.average(test_epoch_loss)))
        print('The Mean Squared Error of forecasts is {} (raw)'.format(np.average(test_epoch_mse)))
        print('The Mean Absolute Error of forecasts is {} (raw)'.format(np.average(test_epoch_mae)))

        if args.show_plot:
            for p_id in range(72):
                draw_true_pre_compare_normal(xscaler.inverse_transform(
                    Xtest.reshape(-1, num_features)).reshape(X_test_all.shape[0], -1, num_features)[p_id, :, 0],
                                      yPred_test[p_id], ytest[p_id], p_id)



if __name__ == "__main__":
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
    parser.add_argument("--d_model", "-dm", type=int, default=128)  # 嵌入维度
    parser.add_argument("--nhead", "-nh", type=int, default=8)  # 注意力头数量
    parser.add_argument("--dim_feedforward", "-hs", type=int, default=256)
    parser.add_argument("--dec_seq_len", "-dl", type=int, default=12)  # decoder用到的输入长度
    parser.add_argument("--out_seq_len", "-ol", type=int, default=24)  # 预测长度
    parser.add_argument("--enc_seq_len", "-not", type=int, default=24*2)  # 输入训练长度
    parser.add_argument("-dropout", type=float, default=0.1)
    parser.add_argument("-activation", type=str, default='relu')

    parser.add_argument("--run_test", "-rt", action="store_true", default=True)
    parser.add_argument("--save_model", "-sm", type=bool, default=True)
    parser.add_argument("--load_model", "-lm", type=bool, default=False)
    parser.add_argument("--show_plot", "-sp", type=bool, default=True)

    parser.add_argument("--day_periods", "-dp", type=int, default=288)
    parser.add_argument("--num_periods", "-np", type=int, default=24)
    parser.add_argument("--num_days", "-ds", type=int, default=30)

    args = parser.parse_args()

    if args.run_test:
        X_all = pickle.load(open(r"../../data/ECW_newmac.pkl", 'rb'))
        y_all = X_all[:, :, 0]
        reference(X_all[:, :, :4], y_all, args)