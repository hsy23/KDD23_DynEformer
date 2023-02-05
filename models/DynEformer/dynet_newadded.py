import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
import argparse
import math
from dyneformerV1 import DynEformer
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x


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
        for j in range(num_obs_to_train, num_periods - seq_len, step):
            X_all.append(X[i, j-num_obs_to_train:j+seq_len, :])
            Y_all.append(Y[i, j:j+seq_len])

    X_all = np.asarray(X_all).reshape(-1, num_obs_to_train+seq_len, n_feats)
    Y_all = np.asarray(Y_all).reshape(-1, seq_len)
    return X_all, Y_all



def reference(X, y, args):
    # model = pickle.load(open("saved_model/GpsFormer_01141318.pkl", 'rb'))
    model = torch.load('saved_model/GPS_ppio_best.pt')
    device = torch.device('cuda:0')

    num_ts, num_periods, num_features = X.shape
    input_size = 4  # encoder输入维度 只包括bw和日期mark
    with open(r'../GlobalPooing/vade_pooling/global_pool_c551_s48_s12.pkl', 'rb') as f:
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

        # test_epoch_loss.append(mse(yPred_test, yscaler.transform(ytest_tensor.cpu().reshape(-1, 1)).to(device)).item())
        yPred_test = yPred_test.cpu().numpy()

        if yscaler is not None:
            yPred_test = yscaler.inverse_transform(yPred_test.reshape(-1, 1))

        test_epoch_mse.append(((ytest.reshape(-1) - yPred_test.reshape(-1)) ** 2).mean())
        test_epoch_mae.append(np.abs(ytest.reshape(-1) - yPred_test.reshape(-1)).mean())
        # pickle.dump([xscaler.inverse_transform(Xtest.reshape(-1, num_features)).reshape(X_test_all.shape[0], -1, num_features), yPred_test, ytest],
        #             open('../../draw_pics/draw_new/gpsformer_newapp_plot.pkl', 'wb'))

        # print('The Test MSE Loss is {}'.format(np.average(test_epoch_loss)))
        print('The Mean Squared Error of forecasts is {} (raw)'.format(np.average(test_epoch_mse)))
        print('The Mean Absolute Error of forecasts is {} (raw)'.format(np.average(test_epoch_mae)))

        # if args.show_plot:
        #     for p_id in range(72):
        #         draw_true_pre_compare(xscaler.inverse_transform(
        #             Xtest.reshape(-1, num_features)).reshape(X_test_all.shape[0], -1, num_features)[p_id, :, 0],
        #                               yPred_test[p_id], ytest[p_id], p_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard_scaler", "-ss", action="store_true")
    parser.add_argument("--log_scaler", "-ls", action="store_true")
    parser.add_argument("--mean_scaler", "-ms", action="store_true")
    parser.add_argument("--minmax_scaler", "-mm", action="store_true", default=True)

    parser.add_argument("--num_epoches", "-e", type=int, default=300)
    parser.add_argument("--step_per_epoch", "-spe", type=int, default=3)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", "-b", type=int, default=6)
    parser.add_argument("--step", "-stp", type=int, default=12)

    parser.add_argument("--n_encoder_layers", "-nel", type=int, default=2)
    parser.add_argument("--n_decoder_layers", "-ndl", type=int, default=1)
    parser.add_argument("--d_model", "-dm", type=int, default=256)  # 嵌入维度
    parser.add_argument("--nhead", "-nh", type=int, default=8)  # 注意力头数量
    parser.add_argument("--dim_feedforward", "-hs", type=int, default=512)
    parser.add_argument("--dec_seq_len", "-dl", type=int, default=12)  # decoder用到的输入长度
    parser.add_argument("--out_seq_len", "-ol", type=int, default=24)  # 预测长度
    parser.add_argument("--enc_seq_len", "-not", type=int, default=24*2)  # 输入训练长度
    parser.add_argument("-dropout", type=float, default=0.5)
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
        # old_data_path = get_data_path("merged_0801_0830_bd_t_feats_hour.pkl")
        # data_path = get_data_path("merged_0901_0930_bd_t_feats_hour.pkl")
        # mac_attr_path = get_data_path("20220901-20220930_mac_attr.pkl")
        #
        # old_data = pd.read_pickle(open(old_data_path, 'rb'))
        # data = pd.read_pickle(open(data_path, 'rb'))
        # mac_attr = pd.read_pickle(open(mac_attr_path, 'rb'))
        #
        # old_app = old_data['name'].unique()
        # data_app = data[data['name'].apply(lambda x: x not in old_app and x != 'default' and x!= 'ipaasDetectd')]
        #
        # data_with_mac_feats = mac_attributes_pro(mac_attr, data_app)
        # # data_with_mac_feats = data_with_mac_feats[data_with_mac_feats['name']=='dcache']
        # X_all = []
        # y_all = []
        #
        # features = ["bw_upload", "hour", "day", "week", 'province', 'bandwidth_type', 'nat_type', 'isp', 'billing_rule',
        #           'upbandwidth', 'upbandwidth_base', 'cpu_num', 'memory_size', 'disk_size', 'test_sat', 'loss_sat']
        # num_feats = len(features)
        # macs = data_with_mac_feats['machine_id'].unique()
        #
        # for mac in macs:
        #     s = data_with_mac_feats[data_with_mac_feats['machine_id'] == mac]
        #     if len(s) != args.num_periods*30:
        #         continue
        #     X = []
        #     y = []
        #
        #     X.append(s[features].values)
        #     y.append(s['bw_upload'])
        #
        #     X_all.append(X)
        #     y_all.append(y)
        #
        # X_all = np.asarray(X_all).reshape((-1, args.num_periods*30, num_feats))
        # y_all = np.asarray(y_all).reshape((-1, args.num_periods*30))

        #pickle.dump(X_all, open(r"../../../raw_data/x_newapp.pkl", 'wb'))
        #pickle.dump(y_all, open(r"../../../raw_data/y_newapp.pkl", 'wb'))

        X_all = pickle.load(open(r"../../../raw_data/x_newmac.pkl", 'rb'))
        y_all = pickle.load(open(r"../../../raw_data/y_newmac.pkl", 'rb'))

        reference(X_all, y_all, args)
