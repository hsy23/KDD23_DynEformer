#!/usr/bin/python 3.6
# -*-coding:utf-8-*-

'''
DeepAR Model (Pytorch Implementation)
Paper Link: https://arxiv.org/abs/1704.04110
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
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
import util
from datetime import date
import argparse
from tqdm import tqdm
from models.data_process_utils import *


class Gaussian(nn.Module):

    def __init__(self, hidden_size, output_size):
        '''
        Gaussian Likelihood Supports Continuous Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(Gaussian, self).__init__()
        self.mu_layer = nn.Linear(hidden_size, output_size)
        self.sigma_layer = nn.Linear(hidden_size, output_size)

        # initialize weights
        # nn.init.xavier_uniform_(self.mu_layer.weight)
        # nn.init.xavier_uniform_(self.sigma_layer.weight)

    def forward(self, h):
        _, hidden_size = h.size()
        sigma_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-6
        sigma_t = sigma_t.squeeze(0)
        mu_t = self.mu_layer(h).squeeze(0)
        return mu_t, sigma_t


class NegativeBinomial(nn.Module):

    def __init__(self, input_size, output_size):
        '''
        Negative Binomial Supports Positive Count Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(NegativeBinomial, self).__init__()
        self.mu_layer = nn.Linear(input_size, output_size)
        self.sigma_layer = nn.Linear(input_size, output_size)

    def forward(self, h):
        _, hidden_size = h.size()
        alpha_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-6
        mu_t = torch.log(1 + torch.exp(self.mu_layer(h)))
        return mu_t, alpha_t


def gaussian_sample(mu, sigma):
    '''
    Gaussian Sample
    Args:
    ytrue (array like)
    mu (array like)
    sigma (array like): standard deviation

    gaussian maximum likelihood using log
        l_{G} (z|mu, sigma) = (2 * pi * sigma^2)^(-0.5) * exp(- (z - mu)^2 / (2 * sigma^2))
    '''
    # likelihood = (2 * np.pi * sigma ** 2) ** (-0.5) * \
    #         torch.exp((- (ytrue - mu) ** 2) / (2 * sigma ** 2))
    # return likelihood
    gaussian = torch.distributions.normal.Normal(mu, sigma)
    # ypred = gaussian.sample(mu.size())
    ypred = gaussian.sample()
    return ypred


def negative_binomial_sample(mu, alpha):
    '''
    Negative Binomial Sample
    Args:
    ytrue (array like)
    mu (array like)
    alpha (array like)

    maximuze log l_{nb} = log Gamma(z + 1/alpha) - log Gamma(z + 1) - log Gamma(1 / alpha)
                - 1 / alpha * log (1 + alpha * mu) + z * log (alpha * mu / (1 + alpha * mu))

    minimize loss = - log l_{nb}

    Note: torch.lgamma: log Gamma function
    '''
    var = mu + mu * mu * alpha
    ypred = mu + torch.randn(mu.size()) * torch.sqrt(var)
    return ypred


class DeepAR(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, lr=1e-3, likelihood="g"):
        super(DeepAR, self).__init__()

        # network
        self.input_embed = nn.Linear(1, embedding_size)
        self.encoder = nn.LSTM(embedding_size + input_size, hidden_size, \
                               num_layers, bias=True, batch_first=True)
        if likelihood == "g":
            self.likelihood_layer = Gaussian(hidden_size, 1)
        elif likelihood == "nb":
            self.likelihood_layer = NegativeBinomial(hidden_size, 1)
        self.likelihood = likelihood

    def forward(self, X, y, Xf):
        '''
        Args:
        X (array like): shape (num_time_series, seq_len, input_size)
        y (array like): shape (num_time_series, seq_len)
        Xf (array like): shape (num_time_series, horizon, input_size)
        Return:
        mu (array like): shape (batch_size, seq_len)
        sigma (array like): shape (batch_size, seq_len)
        '''
        if isinstance(X, type(np.empty(2))):
            X = torch.from_numpy(X).float()
            y = torch.from_numpy(y).float()
            Xf = torch.from_numpy(Xf).float()
        num_ts, seq_len, _ = X.size()
        _, output_horizon, num_features = Xf.size()
        ynext = None
        ypred = []
        mus = []
        sigmas = []
        h, c = None, None
        for s in range(seq_len + output_horizon):
            if s < seq_len:
                ynext = y[:, s].view(-1, 1)
                yembed = self.input_embed(ynext).view(num_ts, -1)
                x = X[:, s, :].view(num_ts, -1)
            else:
                yembed = self.input_embed(ynext).view(num_ts, -1)
                x = Xf[:, s - seq_len, :].view(num_ts, -1)
            x = torch.cat([x, yembed], dim=1)  # num_ts, num_features + embedding
            inp = x.unsqueeze(1)
            # print("inp shape:{}".format(inp.shape))
            if h is None and c is None:
                out, (h, c) = self.encoder(inp)  # h size (num_layers, num_ts, hidden_size)
            else:
                out, (h, c) = self.encoder(inp, (h, c))
            hs = h[-1, :, :]
            hs = F.relu(hs)
            mu, sigma = self.likelihood_layer(hs)
            mus.append(mu.view(-1, 1))
            sigmas.append(sigma.view(-1, 1))
            # print('ynext shape1:{}'.format(ynext.shape))
            if self.likelihood == "g":
                ynext = gaussian_sample(mu, sigma)
            elif self.likelihood == "nb":
                alpha_t = sigma
                mu_t = mu
                ynext = negative_binomial_sample(mu_t, alpha_t)
            # print('ynext shape2:{}'.format(ynext.shape))
            # if without true value, use prediction
            if s >= seq_len - 1 and s < output_horizon + seq_len - 1:
                ypred.append(ynext)
        ypred = torch.cat(ypred, dim=1).view(num_ts, -1)
        mu = torch.cat(mus, dim=1).view(num_ts, -1)
        sigma = torch.cat(sigmas, dim=1).view(num_ts, -1)
        return ypred, mu, sigma


def batch_generator(X, y, num_obs_to_train, seq_len, batch_size):
    '''
    Args:
    X (array like): shape (num_samples, num_features, num_periods)
    y (array like): shape (num_samples, num_periods)
    num_obs_to_train (int):
    seq_len (int): sequence/encoder/decoder length
    batch_size (int)
    '''
    num_ts, num_periods, _ = X.shape
    if num_ts < batch_size:  # 序列数
        batch_size = num_ts
    t = random.choice(range(num_obs_to_train, num_periods - seq_len))  # 一次输入的截断数据下标t前为预测前输入 t后为被预测序列输入
    batch = random.sample(range(num_ts), batch_size)
    X_train_batch = X[batch, t - num_obs_to_train:t, :]
    y_train_batch = y[batch, t - num_obs_to_train:t]
    Xf = X[batch, t:t + seq_len]
    yf = y[batch, t:t + seq_len]
    return X_train_batch, y_train_batch, Xf, yf


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

    X_train_all = np.asarray(X_train_all).reshape(-1, num_obs_to_train, n_feats)
    Y_train_all = np.asarray(Y_train_all).reshape(-1, num_obs_to_train)
    Xf_all = np.asarray(Xf_all).reshape(-1, seq_len, n_feats)
    Yf_all = np.asarray(Yf_all).reshape(-1, seq_len)
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
    Xtr, ytr, Xte, yte = util.train_test_split(X, y)

    if args.likelihood == "g":
        loss_func = util.gaussian_likelihood_loss
    elif args.likelihood == "nb":
        loss_func = util.negative_binomial_loss

    losses = []
    mse_loss = nn.MSELoss().to(device)
    cnt = 0

    yscaler = None
    if args.standard_scaler:
        yscaler = util.StandardScaler()
    elif args.log_scaler:
        yscaler = util.LogScaler()
    elif args.mean_scaler:
        yscaler = util.MeanScaler()
    if yscaler is not None:
        ytr = yscaler.fit_transform(ytr)

    # training
    seq_len = args.seq_len
    num_obs_to_train = args.num_obs_to_train
    X_train_all, Y_train_all, Xf_all, Yf_all = batch_generator_all(Xtr, ytr, num_obs_to_train, seq_len)
    for epoch in tqdm(range(args.num_epoches)):
        # print("Epoch {} starts...".format(epoch))
        for step in range(int(len(X_train_all)/args.batch_size)):
            # Xtrain, ytrain, Xf, yf = batch_generator(Xtr, ytr, num_obs_to_train, seq_len, args.batch_size)
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
            if (epoch % 5 == 0 and step == 0):
                print('The MSE Loss {}'.format(mse_l.item()))
                print("gaussian loss:{}".format(loss.item()))

            losses.append(mse_l.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1

    model = model.cpu()
    pickle.dump(model, open("deepar_{}.pkl".format(time.strftime("%m%d%H%M", time.localtime())), 'wb'))
    # test
    mape_list = []
    # select skus with most top K
    X_test = Xte[:, -seq_len - num_obs_to_train:-seq_len, :].reshape((num_ts, -1, num_features))
    Xf_test = Xte[:, -seq_len:, :].reshape((num_ts, -1, num_features))
    y_test = yte[:, -seq_len - num_obs_to_train:-seq_len].reshape((num_ts, -1))
    yf_test = yte[:, -seq_len:].reshape((num_ts, -1))
    if yscaler is not None:
        y_test = yscaler.transform(y_test)
    result = []
    n_samples = args.sample_size
    for _ in tqdm(range(n_samples)):
        y_pred, _, _ = model(X_test, y_test, Xf_test)
        y_pred = y_pred.data.numpy()
        if yscaler is not None:
            y_pred = yscaler.inverse_transform(y_pred)
        result.append(y_pred.reshape((-1, 1)))

    result = np.concatenate(result, axis=1)  # reshape len(test_num)*list[10] to 100*10
    p50 = np.quantile(result, 0.5, axis=1)
    p90 = np.quantile(result, 0.9, axis=1)
    p10 = np.quantile(result, 0.1, axis=1)

    mape = util.MAPE(yf_test, p50)
    print("P50 MAPE: {}".format(mape))
    mape_list.append(mape)

    print('The Mean Squared Error of forecasts is {}'.format(((p50 - yf_test.reshape(-1)) ** 2).mean()))
    print('The Root Mean Squared Error of forecasts is {}'.format(np.sqrt(((p50 - yf_test.reshape(-1)) ** 2).mean())))

    # if args.show_plot:
    #     plt.figure(1, figsize=(20, 5))
    #     plt.plot([len(y[0]) - k for k in range(seq_len, 0, -1)], p50, "r-")
    #     plt.fill_between(x=[len(y[0]) - k for k in range(seq_len, 0, -1)], y1=p10, y2=p90, alpha=0.5)
    #     plt.title('Prediction uncertainty')
    #     # yplot = yte[-1, -seq_len - num_obs_to_train:]
    #     # plt.plot(range(len(yplot)), yplot, "k-")
    #     plt.plot(range(len(y[0])), y[0], "k-")
    #     plt.legend(["P50 forecast", "true", "P10-P90 quantile"], loc="upper left")
    #     ymin, ymax = plt.ylim()
    #     plt.vlines(len(y[0]) - seq_len, ymin, ymax, color="blue", linestyles="dashed", linewidth=2)
    #     plt.ylim(ymin, ymax)
    #     plt.xlabel("Periods")
    #     plt.ylabel("Y")
    #     plt.show()
    return losses, mape_list


def Pre_V5_more_feats(df, test_dt):  # test_dt用于计算时间距离以删除数据 min_test_dt用于保留被测试数据
    df = dt_2_dayl(df)  # 将日期转为时间数组
    index_used = ['device_uuid', 'dbi_server_type', 'task_id', 'province',
                  'city', 'bandwidth_type', 'bd_name', 'upbandwidth', 'upbandwidth_perline',
                  'nat_type', 'isp', 'cpu_num', 'cpu_type', 'cpu_frequency', 'memory_size', 'dbi_memory_type',
                  'disk_size', 'billing_rule', 'hdd', 'hdd_num', 'ssd', 'ssd_num', 'nvme', 'nvme_num',
                  'billing_time', 'test_sat', 'loss_sat', 'isin_req', 'day', 'week', 'mon', 'use_rate', 'dt']
    features = index_used[1:-1]
    index_pro = ['use_rate']
    feats_num = ['billing_time',
                'upbandwidth', 'upbandwidth_perline',
                'cpu_num', 'cpu_frequency',
                'memory_size',
                'disk_size', 'hdd', 'hdd_num', 'ssd', 'ssd_num', 'nvme', 'nvme_num',
                'test_sat', 'loss_sat', 'use_rate', 'dt']
    feats_cate = ['task_id', 'new_task_id', 'dbi_server_type', 'province', 'city', 'bandwidth_type', 'bd_name',
                  'nat_type', 'isp', 'cpu_type', 'dbi_memory_type', 'billing_rule', 'isin_req', 'day', 'week',
                  'mon']

    df_new = df[index_used].copy()
    df_new = df_new.sort_values(by=['device_uuid', 'dt']).reset_index().iloc[:, 1:]
    # Todo:识别是否切任务0/1是否切任务 5->5/7（新任务）
    # Todo: 任务的feature embedding
    # 通过排序后数据交错，将下一天的use_rate作为前一天的label
    df_new['label'] = np.nan
    df_new['label'].iloc[:-1] = df_new['use_rate'].iloc[1:]

    # 新增任务切换属性
    df_new['new_task_id'] = np.nan
    df_new['new_task_id'].iloc[:-1] = df_new['task_id'].iloc[1:]
    df_new['task_change'] = np.nan
    df_new['task_change'] = df_new[['task_id', 'new_task_id']].apply(lambda x: 0 if x[0] == x[1] else 1, axis=1)

    # 删除每台机器最后一天的数据，因为其label来自不同机器
    df_new = df_new.groupby(['device_uuid']).apply(lambda x: x.iloc[:-1])

    # df_new = df_new[df_new['task_change'] == 1]

    df_new.loc[:, feats_num] = df_new.loc[:, feats_num].astype(float)
    df_new.loc[:, feats_cate] = df_new.loc[:, feats_cate].astype(str)

    df_new = delete_history_mac(df_new, test_dt)
    df_new = delete_abnoraml_mac(df_new)
    df_all = df_new

    data_X, data_Y = df_all[features+['new_task_id', 'task_change']], df_all['label']
    # borderline = len(df_all[df_all['dt'].apply(lambda x: x < int(test_dt))])
    borderline = int(0.8 * len(df_all))
    train_X, train_Y = data_X.iloc[:borderline, :], data_Y.iloc[:borderline]
    test_X, test_Y = data_X.iloc[borderline:, :], data_Y.iloc[borderline:]

    train_X, test_X, scalers = std_data(train_X, test_X, cols=feats_num, way='std')  # 连续数据标准化后的scaler
    train_X, test_X, encode_dicts, std_scalers = \
        discrete_feats_encoder(train_X, test_X, train_Y, feats_cate, encoding_way='target encoding', std='minmax')

    mlp_model = deploy_mlp_2(train_X.values, train_Y.values, test_X.values, test_Y.values)
    return mlp_model, scalers, encode_dicts, std_scalers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoches", "-e", type=int, default=50)
    parser.add_argument("--step_per_epoch", "-spe", type=int, default=3)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("--n_layers", "-nl", type=int, default=3)
    parser.add_argument("--hidden_size", "-hs", type=int, default=64)
    parser.add_argument("--embedding_size", "-es", type=int, default=64)
    parser.add_argument("--likelihood", "-l", type=str, default="g")
    parser.add_argument("--seq_len", "-sl", type=int, default=24)  # 预测长度
    parser.add_argument("--num_obs_to_train", "-not", type=int, default=24*2)  # 输入训练长度
    parser.add_argument("--num_results_to_sample", "-nrs", type=int, default=10)
    parser.add_argument("--show_plot", "-sp", action="store_true", default=True)
    parser.add_argument("--run_test", "-rt", action="store_true", default=True)
    parser.add_argument("--standard_scaler", "-ss", action="store_true")
    parser.add_argument("--log_scaler", "-ls", action="store_true")
    parser.add_argument("--mean_scaler", "-ms", action="store_true", default=True)
    parser.add_argument("--batch_size", "-b", type=int, default=256)

    parser.add_argument("--sample_size", type=int, default=10)  # 为每个预测生成的候选数量
    parser.add_argument("--day_periods", "-per", type=int, default=288)
    parser.add_argument("--num_days", "-ds", type=int, default=30)

    args = parser.parse_args()

    if args.run_test:
        # data_path = util.get_data_path("pro_0801_0830_bd.pkl")
        # mac_attr_path = util.get_data_path("20220801-20220830_mac_attr.pkl")
        # mac_pro_path = util.get_data_path("20220801-20220830_mac_pro.pkl")
        # demand_area_path = util.get_data_path("20220830_demand_area.pkl")
        #
        # data = pd.read_pickle(open(data_path, 'rb'))
        # # mac_attr = pd.read_pickle(open(mac_attr_path, 'rb'))
        # # mac_pro = pd.read_pickle(open(mac_pro_path, 'rb'))
        # # demand_area = pd.read_pickle(open(demand_area_path, 'rb'))
        # # data = app_filter(data)
        # data = data.sort_values(by=['machine_id', 'time_id'])
        # #
        # data = dt_2_dayl(data)
        # data['bw_upload'] = data['bw_upload']/1024/1024/1024  # GB
        # # data = data.loc[(data["date"].dt.date >= date(2022, 8, 25)) & (data["date"].dt.date <= date(2014, 3, 1))]
        # # add_feats = data_pro_merge(mac_attr)
        # X_all = []
        # y_all = []
        #
        # num_periods = args.num_days*args.day_periods
        # features = ["second", "hour", "day", "week", "mon"]
        # num_feats = len(features)
        # # 保留满足完整时序的数据
        # # tmp = data.groupby(['machine_id', 'name']).size().reset_index()
        # # tmp = data.groupby('machine_id').size().reset_index()
        # # tmp2 = tmp[tmp[0] == num_periods]
        # # tmp3 = pd.merge(data, tmp2, on=['machine_id', 'name']).iloc[:, :-1]
        # # tmp3 = pd.merge(data, tmp2, on=['machine_id']).iloc[:, :-1]
        #
        # for i in range(0, len(data), num_periods):
        #     tmp4 = data.iloc[i:i+num_periods, :]
        #     # if len(tmp4['machine_id'].unique()) != len(tmp4['name'].unique()) or len(tmp4['machine_id'].unique()) != 1:
        #     #     raise ValueError("出现新mac或新task")
        #     if len(tmp4['machine_id'].unique()) != 1:
        #         raise ValueError("出现新mac")
        #     X = tmp4[features].values
        #     X = np.asarray(X).reshape((num_periods, num_feats))  # num_series
        #     y = np.asarray(tmp4["bw_upload"].astype(float)).reshape((num_periods))
        #
        #     X_all.append(X)
        #     y_all.append(y)
        #
        # X_all = np.asarray(X_all).reshape((-1, num_periods, num_feats))
        # y_all = np.asarray(y_all).reshape((-1, num_periods))
        # losses, mape_list = train(X_all, y_all, args)
        # if args.show_plot:
        #     plt.plot(range(len(losses)), losses, "k-")
        #     plt.xlabel("Period")
        #     plt.ylabel("Loss")
        #     plt.show()
        X_all = pickle.load(open(get_data_path("X_all_0801_0830.pkl"), 'rb'))
        y_all = pickle.load(open(get_data_path("y_all_0801_0830.pkl"), 'rb'))
        X_all = X_all[:, :, 1:]  # 去掉特征中的y元素
        losses, mape_list = train(X_all, y_all, args)
        pickle.dump(losses, open("losses_{}.pkl".format(time.strftime("%m%d%H%M", time.localtime())), 'wb'))
        if args.show_plot:
            plt.plot(range(len(losses)), losses, "k-")
            plt.xlabel("Period")
            plt.ylabel("Loss")
            plt.show()
