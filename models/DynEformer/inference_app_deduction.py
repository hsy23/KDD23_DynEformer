import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
import argparse
import math
from dyneformer_tune import DynEformer
from sklearn import preprocessing
from tqdm import tqdm
from torch.optim import Adam
import torch
import time
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.append('../')
sys.path.append('../GlobalPooing')  # 添加路径以找到类

from data_process_utils import *
from global_utils import *
from use_cases_utils import get_app_deduction


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


def batch_generator_padding(X, Y, num_obs_to_train, seq_len, step):
    '''
        Args:
        X (array like): shape (num_samples, num_features, num_periods)
        y (array like): shape (num_samples, num_periods)
        num_obs_to_train (int):
        seq_len (int): sequence/encoder/decoder length
        batch_size (int)
        '''
    num_ts, num_periods, n_feats = X.shape
    X_all = []
    Y_all = []

    for i in range(num_ts):
        for j in range(num_obs_to_train, num_periods - seq_len + 1, step):
            X_all.append(X[i, j-num_obs_to_train:j+seq_len, :])
            Y_all.append(Y[i, j:j+seq_len])

    X_all = np.asarray(X_all).reshape(-1, num_obs_to_train+seq_len, n_feats)
    Y_all = np.asarray(Y_all).reshape(-1, seq_len)
    return X_all, Y_all


def inference(X, y, args):
    model = torch.load('saved_model/Dyneformer_best.pt')
    device = torch.device('cuda:0')

    num_ts, num_periods, num_features = X.shape
    input_size = 4  # encoder输入维度 只包括bw和日期mark
    with open(r'../GlobalPooing/vade_search_cluster/global_pool_c551_s48_s12.pkl', 'rb') as f:
        timeFeature_pool = pickle.load(f)
    sFeature = torch.tensor(np.stack(timeFeature_pool.seasonal_pool).T).to(device)
    timeFeature = sFeature
    timeFeature_len = timeFeature.shape[-1]
    model = model.to(device)

    random.seed(2)
    mse = nn.MSELoss().to(device)

    Xte = X
    yte = y

    xscaler, yscaler = pickle.load(open('8_scalers.pkl', 'rb'))
    Xte = xscaler.transform(Xte.reshape(-1, num_features)).reshape(num_ts, -1, num_features)

    pre_seq_len = args.out_seq_len
    num_obs_to_train = args.enc_seq_len
    X_test_all, Y_test_all = batch_generator_padding(Xte, yte, num_obs_to_train, pre_seq_len, args.step)

    with torch.no_grad():
        test_epoch_loss = []
        test_epoch_mse = []
        test_epoch_mae = []

        Xtest, ytest = X_test_all, Y_test_all

        Xtest_tensor = torch.from_numpy(Xtest.astype(float)).float().to(device)[:, :, :4]
        Test_static_context = torch.from_numpy(Xtest.astype(float)).float().to(device)[:, 0, 4:].squeeze(1)
        ytest_tensor = torch.from_numpy(ytest.astype(float)).float().to(device)

        yPred_test = model(Xtest_tensor, Test_static_context, timeFeature)

        test_epoch_loss.append(mse(yPred_test, yscaler.transform(ytest_tensor)).item())
        yPred_test = yPred_test.cpu().numpy()

        # if args.show_plot:
        #     for p_id in range(72):
        #         draw_true_pre_compare(Xtest[p_id, :, 0], yPred_test[p_id], yscaler.transform(ytest)[p_id], p_id)

        if yscaler is not None:
            yPred_test = yscaler.inverse_transform(yPred_test)

        test_epoch_mse.append(((ytest.reshape(-1) - yPred_test.reshape(-1)) ** 2).mean())
        test_epoch_mae.append(np.abs(ytest.reshape(-1) - yPred_test.reshape(-1)).mean())
        label_deduction = get_app_deduction(ytest)
        pre_deduction = get_app_deduction(yPred_test)
        # pickle.dump([xscaler.inverse_transform(Xtest.reshape(-1, num_features)).reshape(X_test_all.shape[0], -1, num_features), yPred_test, ytest],
        #             open('gpsformer_switch_plot.pkl', 'wb'))

        print('The Test MSE Loss is {}'.format(np.average(test_epoch_loss)))
        print('The Mean Squared Error of forecasts is {} (raw)'.format(np.average(test_epoch_mse)))
        print('The Mean Absolute Error of forecasts is {} (raw)'.format(np.average(test_epoch_mae)))
        print('label deduction{}, pre deduction{}'.format(label_deduction, pre_deduction))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard_scaler", "-ss", action="store_true")
    parser.add_argument("--log_scaler", "-ls", action="store_true")
    parser.add_argument("--mean_scaler", "-ms", action="store_true")
    parser.add_argument("--minmax_scaler", "-mm", action="store_true", default=True)

    parser.add_argument("--num_epoches", "-e", type=int, default=300)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", "-b", type=int, default=256)
    parser.add_argument("--step", type=int, default=12)

    parser.add_argument("--n_encoder_layers", "-nel", type=int, default=2)
    parser.add_argument("--n_decoder_layers", "-ndl", type=int, default=1)
    parser.add_argument("--d_model", "-dm", type=int, default=256)  # 嵌入维度
    parser.add_argument("--nhead", "-nh", type=int, default=8)  # 注意力头数量
    parser.add_argument("--dim_feedforward", "-hs", type=int, default=512)
    parser.add_argument("--dec_seq_len", "-dl", type=int, default=12)  # decoder用到的输入长度
    parser.add_argument("--out_seq_len", "-ol", type=int, default=24)  # 预测长度
    parser.add_argument("--enc_seq_len", "-not", type=int, default=24*2)  # 输入训练长度
    parser.add_argument("-dropout", type=float, default=0.1)
    parser.add_argument("-activation", type=str, default='relu')

    parser.add_argument("--run_test", "-rt", action="store_true", default=True)
    parser.add_argument("--save_model", "-sm", type=bool, default=True)
    parser.add_argument("--load_model", "-lm", type=bool, default=False)
    parser.add_argument("--show_plot", "-sp", type=bool, default=False)

    parser.add_argument("--day_periods", "-dp", type=int, default=288)
    parser.add_argument("--num_periods", "-np", type=int, default=24)
    parser.add_argument("--num_days", "-ds", type=int, default=30)

    args = parser.parse_args()

    if args.run_test:
        data_path = get_data_path("merged_0801_0830_bd_t_feats_hour.pkl")
        mac_attr_path = get_data_path("20220801-20220830_mac_attr.pkl")

        data = pd.read_pickle(open(data_path, 'rb'))
        mac_attr = pd.read_pickle(open(mac_attr_path, 'rb'))

        data_with_mac_feats = mac_attributes_pro(mac_attr, data)
        data_app = data_with_mac_feats[data_with_mac_feats['task_id'] == 0]  # 0:kuaishou 1:bilibili 2tencent 7huya 15zjtd
        data_used = data_app[(data_app['dt']>'20220825') & (data_app['dt']<'20220829')]

        X_all = []
        y_all = []

        features = ["bw_upload", "hour", "day", "week", 'province', 'bandwidth_type', 'nat_type', 'isp', 'billing_rule',
                  'upbandwidth', 'upbandwidth_base', 'cpu_num', 'memory_size', 'disk_size', 'test_sat', 'loss_sat']
        num_feats = len(features)

        for i in range(0, len(data_used), args.num_periods*3):
            s = data_used.iloc[i:i + args.num_periods*3, :]
            # if len(s['name'].unique()) != 1:
            #     changed_macs.add(s['machine_id'].values[0])
            assert len(s['machine_id'].unique()) == 1
            X = []
            y = []

            X.append(s[features].values)
            y.append(s['bw_upload'])
            X = np.asarray(X).reshape((args.num_periods*3, num_feats))  # num_series
            y = np.asarray(y).reshape((args.num_periods*3))

            X_all.append(X)
            y_all.append(y)

        X_all = np.asarray(X_all).reshape((-1, args.num_periods*3, num_feats))
        y_all = np.asarray(y_all).reshape((-1, args.num_periods*3))
        inference(X_all, y_all, args)
