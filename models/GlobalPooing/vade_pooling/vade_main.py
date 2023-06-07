import argparse
import pickle

import matplotlib.pyplot as plt
import torch

from dataloader import *
from model_vade import VaDE
from tqdm import tqdm
import numpy as np
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR
from scipy.optimize import linear_sum_assignment as linear_assignment
from build_global_pool import build_global_pool
from sklearn.manifold import TSNE
import torch.nn as nn


def cluster_acc(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind])*1.0/Y_pred.size, w


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='VaDE')
    parse.add_argument('--batch_size', type=int, default=40000)
    parse.add_argument('--dataX_dir', type=str, default='../../data/ECW_08.pkl')
    parse.add_argument('--nClusters', type=int, default=400)
    # parse.add_argument('--step_per_epoch', type=int, default=300)

    parse.add_argument('--series_len', type=int, default=48)
    parse.add_argument('--step', type=int, default=12)

    parse.add_argument('--hid_dim', type=int, default=10)
    parse.add_argument('--cuda', type=bool, default=True)

    args=parse.parse_args()

    X = pickle.load(open(args.dataX_dir, 'rb'))
    X = np.asarray(X)
    X_train = X[:, :24*25]  # 筛选训练集数据

    vade = VaDE(args, X_train)
    if args.cuda:
        vade = vade.cuda()
        # vade=nn.DataParallel(vade, device_ids=range(4))

    best_n = vade.pre_train(pre_epoch=1000)
    for cn in range(100, 650, 100):
        build_global_pool(cn, args)








