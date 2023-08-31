import pickle
import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset
import random
from sklearn.preprocessing import MinMaxScaler
import sys

sys.path.append('../')
from series_decomp import s_decomp
from tqdm import tqdm


def get_pworkload(X, Y, wtype='train', std=None, series_len=24*2, step=12, batch_size=256):
    '''
        Args:
        X (array like): shape (num_samples, num_features, num_periods)
        y (array like): shape (num_samples, num_periods)
        num_obs_to_train (int):
        seq_len (int): sequence/encoder/decoder length
        batch_size (int)
        '''
    if type(X) == list:
        X = np.asarray(X)
    num_ts, num_periods = X.shape
    new_X = []
    for s in tqdm(X):  # If the data requires additional processing for timing decomposition
        new_X.append(s_decomp(s, type='seasonal'))

    if wtype == 'train':
        scaler = MinMaxScaler()
        new_X = scaler.fit_transform(X.T).T
        pickle.dump(scaler, open('pool_scaler.pkl', 'wb'))
    else:
        scaler = pickle.load(open('pool_scaler.pkl', 'rb'))
        new_X = scaler.transform(X.T).T

    X_train_all = []
    for i in range(num_ts):
        for j in range(series_len, num_periods, step):
            X_train_all.append(new_X[i, j-series_len:j])

    X_train_all = np.asarray(X_train_all).reshape(-1, series_len)
    X_train_all_tensor = torch.from_numpy(X_train_all).float()

    dataloader = DataLoader(TensorDataset(X_train_all_tensor), batch_size=batch_size, shuffle=True, num_workers=4)

    return dataloader, X_train_all_tensor,  X_train_all





