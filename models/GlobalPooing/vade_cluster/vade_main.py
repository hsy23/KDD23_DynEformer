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
    parse.add_argument('--dataX_dir', type=str, default='../../../../raw_data/X_all_0801_0830.pkl')
    parse.add_argument('--nClusters', type=int, default=5)
    parse.add_argument('--step_per_epoch', type=int, default=1000)

    parse.add_argument('--series_len', type=int, default=48)
    parse.add_argument('--step', type=int, default=2)

    parse.add_argument('--hid_dim', type=int, default=10)
    parse.add_argument('--cuda', type=bool, default=True)

    args=parse.parse_args()

    # new_X = []
    # for s in tqdm(X):
    #     new_X.append(s_decomp(s, type=dtype))

    X = pickle.load(open(args.dataX_dir, 'rb'))
    X = np.asarray(X)
    X_train = X[:, :24*25]  # 筛选训练集数据

    vade = VaDE(args, X_train)
    if args.cuda:
        vade = vade.cuda()
        # vade=nn.DataParallel(vade, device_ids=range(4))

    best_n = vade.pre_train(pre_epoch=500)

    # vade = torch.load(open('c5_vade2.pkl', 'rb'))

    x_switch = pickle.load(open('../../../../raw_data/x_switch_0801_0830.pkl', 'rb'))
    x_mac = pickle.load(open('../../../../raw_data/ECW_newmac.pkl', 'rb'))
    x_app = pickle.load(open('../../../../raw_data/ECW_newapp.pkl', 'rb'))
    # vade.predict(X[:, 24*25:], 'switch')
    # vade.predict(x_switch, 'switch')
    vade.predict(x_mac, 'mac')
    # vade.predict(x_app, 'app')
    # build_global_pool(551, args)

    # opti = Adam(vade.parameters(), lr=2e-3)
    # lr_s = StepLR(opti, step_size=10, gamma=0.95)

    # writer = SummaryWriter('./logs')
    #
    # epoch_bar=tqdm(range(10))
    #
    # tsne=TSNE()
    #
    # for epoch in epoch_bar:
    #
    #     lr_s.step()
    #     L = 0
    #     for x in X:
    #         if args.cuda:
    #             x = x.cuda()
    #
    #         loss = vade.module.ELBO_Loss(x)
    #
    #         opti.zero_grad()
    #         loss.backward()
    #         opti.step()
    #
    #         L+=loss.detach().cpu().numpy()
    #
    #
    #     pre=[]
    #     tru=[]
    #
    #     with torch.no_grad():
    #         for x, y in []:
    #             if args.cuda:
    #                 x = x.cuda()
    #
    #             tru.append(y.numpy())
    #             pre.append(vade.module.predict(x))
    #
    #
    #     tru=np.concatenate(tru,0)
    #     pre=np.concatenate(pre,0)
    #
    #
    #     writer.add_scalar('loss',L/len(X),epoch)
    #     writer.add_scalar('acc',cluster_acc(pre,tru)[0]*100,epoch)
    #     writer.add_scalar('lr',lr_s.get_lr()[0],epoch)
    #
    #     epoch_bar.write('Loss={:.4f},ACC={:.4f}%,LR={:.4f}'.format(L/len(DL),cluster_acc(pre,tru)[0]*100,lr_s.get_lr()[0]))








