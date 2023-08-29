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
sys.path.append('../GlobalPooing')

from data_process_utils import *
from global_utils import *


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


class PPIO_Dataset(Dataset):
    def __init__(self, X, enc_len=48, label_len=12, pred_len=24, num_static=12):
        self.pred_len = pred_len
        num_ts, num_periods, num_features = X.shape
        sq_len = enc_len + pred_len
        X_train_all = []
        X.astype(float)

        for i in range(num_ts):
            for j in range(sq_len, num_periods, 12):
                X_train_all.append(X[i, j - sq_len:j, :])

        self.X = np.stack(X_train_all).reshape(-1, sq_len, num_features)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.X[index, -self.pred_len:, 0]


def train(X, y, args):
    device = torch.device('cuda:0')
    input_size = 4
    cn = 551
    print('cn: {}'.format(cn))
    with open(r'../GlobalPooing/vade_search_cluster/global_pool_c{}_s48_s12.pkl'.format(cn), 'rb') as f:
        globalPool_pool = pickle.load(f)
    sFeature = torch.tensor(np.stack(globalPool_pool.seasonal_pool).T).float().to(device)

    globalPool = sFeature
    P = globalPool.shape[-1]

    model = DynEformer(args.d_model, args.d_model, args.dim_feedforward, input_size, args.enc_seq_len, args.dec_seq_len,
                        args.out_seq_len, n_encoder_layers=args.n_encoder_layers, n_decoder_layers=args.n_decoder_layers,
                        n_heads=args.nhead, dropout=args.dropout, activation=args.activation, time_feature_len=P,
                        if_padding=args.if_padding)

    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    random.seed(2)
    # select sku with most top n quantities

    Xtr, ytr, Xte, yte = train_test_split(X, y)
    xscaler = preprocessing.MinMaxScaler()
    yscaler = preprocessing.MinMaxScaler()
    num_ts, num_periods, num_features = Xtr.shape
    Xtr = xscaler.fit_transform(Xtr.reshape(-1, num_features)).reshape(num_ts, num_periods, num_features)
    num_ts, num_periods, num_features = Xte.shape
    Xte = xscaler.transform(Xte.reshape(-1, num_features)).reshape(num_ts, num_periods, num_features)
    yscaler.fit(ytr.reshape(-1, 1))
    Xtr_loader = DataLoader(PPIO_Dataset(Xtr), batch_size=args.batch_size)
    Xte_loader = DataLoader(PPIO_Dataset(Xte), batch_size=args.batch_size)

    # pickle.dump([xscaler, yscaler], open('8_scalers_deeptrans.pkl', 'wb'))

    losses = []
    test_loss = []
    test_mse = []
    test_mae = []
    mse = nn.MSELoss().to(device)

    min_loss = 1000
    # training
    for epoch in tqdm(range(args.num_epoches)):
        train_epoch_loss = []
        for Xtrain, ytrain in Xtr_loader:
            Xtrain_tensor = Xtrain.float().to(device)[:, :, :4]
            Train_static_context = Xtrain.float().to(device)[:, 0, 4:].squeeze(1)
            ytrain_tensor = ytrain.float().to(device)

            ypred = model(Xtrain_tensor, globalPool, Train_static_context)
            loss = mse(ypred, ytrain_tensor)
            train_epoch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(np.average(train_epoch_loss))

        epo_test_losses = []
        epo_mse = []
        epo_mape = []
        epo_mae = []
        model.eval()
        with torch.no_grad():
            for x, y in Xte_loader:
                x_load = x.float().to(device)[:, :, :4]
                x_static = x.float().to(device)[:, 0, 4:].squeeze(1)
                y = y.float().to(device)

                yPred = model(x_load, globalPool, x_static)

                loss = mse(yPred, y)

                epo_test_losses.append(loss.item())
                epo_mse.append(get_mse(yPred.cpu(), y.cpu(), yscaler))
                epo_mape.append(get_mape(yPred.cpu(), y.cpu(), yscaler))
                epo_mae.append(get_mae(yPred.cpu(), y.cpu(), yscaler))

        test_loss.append(np.mean(epo_test_losses))
        test_mse.append(np.mean(epo_mse))
        # test_mape.append(np.mean(epo_mape))
        test_mae.append(np.mean(epo_mae))

        if args.save_model:
            if test_loss[-1] < min_loss:
                best_model = model
                min_loss = test_loss[-1]
                torch.save(model, 'saved_model/DynEformer_best_n{}.pt'.format(cn))

        print(f'epoch {epoch}, train loss: {losses[-1]}, test loss: {test_loss[-1]}, mse: {test_mse[-1]}, mae: {test_mae[-1]}')
    print(np.min(test_mse), np.min(test_mae), np.argmin(test_mse))

    return losses, test_loss, test_mse, test_mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard_scaler", "-ss", action="store_true")
    parser.add_argument("--log_scaler", "-ls", action="store_true")
    parser.add_argument("--mean_scaler", "-ms", action="store_true")
    parser.add_argument("--minmax_scaler", "-mm", action="store_true", default=True)

    parser.add_argument("--num_epoches", "-e", type=int, default=300)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", "-b", type=int, default=256)

    parser.add_argument("--n_encoder_layers", "-nel", type=int, default=2)
    parser.add_argument("--n_decoder_layers", "-ndl", type=int, default=1)
    parser.add_argument("--d_model", "-dm", type=int, default=256)
    parser.add_argument("--nhead", "-nh", type=int, default=8)
    parser.add_argument("--dim_feedforward", "-hs", type=int, default=512)
    parser.add_argument("--dec_seq_len", "-dl", type=int, default=12)
    parser.add_argument("--out_seq_len", "-ol", type=int, default=24)
    parser.add_argument("--enc_seq_len", "-not", type=int, default=24*2)
    parser.add_argument("-dropout", type=float, default=0.5)
    parser.add_argument("-activation", type=str, default='relu')
    parser.add_argument("-if_padding", type=bool, default=True)

    parser.add_argument("--run_test", "-rt", action="store_true", default=True)
    parser.add_argument("--save_model", "-sm", type=bool, default=True)
    parser.add_argument("--load_model", "-lm", type=bool, default=False)
    parser.add_argument("--show_plot", "-sp", type=bool, default=False)

    parser.add_argument("--day_periods", "-dp", type=int, default=288)
    parser.add_argument("--num_periods", "-np", type=int, default=24)
    parser.add_argument("--num_days", "-ds", type=int, default=30)

    args = parser.parse_args()

    if args.run_test:
        X_all = np.load(open(r"../../data/ECW_08.npy", 'rb'), allow_pickle=True)
        y_all = X_all[:, :, 0]
        losses, test_losses, mse_l, mae_l = train(X_all, y_all, args)