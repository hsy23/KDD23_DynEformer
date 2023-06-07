import pickle

import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset
import random
from sklearn.preprocessing import MinMaxScaler
import sys

sys.path.append('../../')
from series_decomp import s_decomp
from tqdm import tqdm


def get_pworkload(X_raw, Y, wtype='train', std=None, series_len=24*2, step=12, batch_size=256):
    '''
        Args:
        X (array like): shape (num_samples, num_features, num_periods)
        y (array like): shape (num_samples, num_periods)
        num_obs_to_train (int):
        seq_len (int): sequence/encoder/decoder length
        batch_size (int)
        '''
    X = X_raw[:, :, 0]
    if type(X) == list:
        X = np.asarray(X)
    num_ts, num_periods = X.shape
    new_X = X
    # new_X = []
    # for s in tqdm(X):  # If the data requires additional processing for timing decomposition
    #     new_X.append(s_decomp(s, type=dtype))

    # if wtype == 'train':
    #     scaler = MinMaxScaler()
    #     new_X = scaler.fit_transform(X.reshape(-1, 1)).reshape(num_ts, num_periods)
    #     pickle.dump(scaler, open('cluster_scaler.pkl', 'wb'))
    # else:
    #     scaler = pickle.load(open('cluster_scaler.pkl', 'rb'))
    #     new_X = scaler.transform(X.reshape(-1, 1)).reshape(num_ts, num_periods)

    X_raw_all = []
    y_raw_all = []
    X_train_all = []
    y_train_all = []
    for i in range(num_ts):
        for j in range(series_len, num_periods-24, 12):
            X_raw_all.append(X_raw[i, j-series_len:j, :])
            y_raw_all.append(X_raw[i, j:j+24, 0])
            X_train_all.append(new_X[i, j-series_len:j])
            y_train_all.append(new_X[i, j:j+24])

    X_raw_all = np.asarray(X_raw_all).reshape(-1, series_len, 16)
    y_raw_all = np.asarray(y_raw_all).reshape(-1, 24)
    X_train_all = np.asarray(X_train_all).reshape(-1, series_len)
    y_train_all = np.asarray(y_train_all).reshape(-1, 24)
    X_train_all_torch = torch.from_numpy(X_train_all.astype(float)).float()

    dataloader = DataLoader(TensorDataset(X_train_all_torch), batch_size=batch_size, shuffle=True, num_workers=4)

    return dataloader, X_train_all_torch,  X_raw_all, y_train_all, y_raw_all





