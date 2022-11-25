import pickle

import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader,TensorDataset
import random
from sklearn.preprocessing import MinMaxScaler
from models.series_decomp import s_decomp
from tqdm import tqdm


def Min_Max(X):
    distance = X.max() - X.min()
    X = (X - X.min()) / distance
    return X


def get_mnist(data_dir='./data/mnist/',batch_size=128):
    train=MNIST(root=data_dir,train=True,download=True)
    test=MNIST(root=data_dir,train=False,download=True)

    X=torch.cat([train.data.float().view(-1,784)/255.,test.data.float().view(-1,784)/255.],0)
    Y=torch.cat([train.targets,test.targets],0)

    dataset=dict()
    dataset['X']=X
    dataset['Y']=Y

    dataloader=DataLoader(TensorDataset(X,Y),batch_size=batch_size,shuffle=True,num_workers=4)

    return dataloader,dataset


def get_pworkload(X, Y, batch_size=128, num_obs_to_train=24*2, seq_len=24):
    '''
        Args:
        X (array like): shape (num_samples, num_features, num_periods)
        y (array like): shape (num_samples, num_periods)
        num_obs_to_train (int):
        seq_len (int): sequence/encoder/decoder length
        batch_size (int)
        '''
    num_ts, num_periods = X.shape
    if num_ts < batch_size:  # 序列数
        batch_size = num_ts
    t = random.choice(range(num_obs_to_train, num_periods - seq_len))  # 一次输入的截断数据下标t前为预测前输入 t后为被预测序列输入
    batch = random.sample(range(num_ts), batch_size)
    X_train_batch = torch.from_numpy(X[batch, t - num_obs_to_train:t]).float()
    y_train_batch = torch.from_numpy(Y[batch, t:t + seq_len]).float()
    return X_train_batch, y_train_batch


def get_pworkload_all(X, Y, dtype='seasonal', std=None, num_obs_to_train=24*2):
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
    # new_X = []
    # for s in tqdm(X):
    #     new_X.append(s_decomp(s, type=dtype))
    #
    # pickle.dump(np.asarray(new_X), open('./X_0801_0830_seasonal.pkl', 'wb'))

    scaler = MinMaxScaler()
    if std == 'minmax':
        new_X = scaler.fit_transform(X.T).T

    new_X2 = Min_Max(X)
    X_train_all = []
    for i in range(num_ts):  # todo:思考是否有必要以滑动方式取曲线
        for j in range(0, num_periods, num_obs_to_train):
            X_train_all.append(new_X[i, j:j+num_obs_to_train])

    X_train_all = np.asarray(X_train_all).reshape(-1, num_obs_to_train)
    X_train_all = torch.from_numpy(X_train_all).float()
    return X_train_all, X_train_all





