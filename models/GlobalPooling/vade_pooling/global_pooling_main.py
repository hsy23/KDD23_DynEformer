import argparse
from dataloader import *
from model_vade import VaDE
import numpy as np
from build_global_pool import build_gp



if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='VaDE')
    parse.add_argument('--batch_size', type=int, default=40000)
    parse.add_argument('--dataX_dir', type=str, default='../../../data/ECW_08.npy')
    parse.add_argument('--nClusters', type=int, default=500)

    parse.add_argument('--series_len', type=int, default=48)
    parse.add_argument('--step', type=int, default=12)

    parse.add_argument('--hid_dim', type=int, default=10)
    parse.add_argument('--cuda', type=bool, default=True)

    args=parse.parse_args()

    X = np.load(open(args.dataX_dir, 'rb'), allow_pickle=True)
    X = np.asarray(X)

    train_len = int(0.8*X.shape[1])
    X_train = X[:, :train_len, 0]

    vade = VaDE(args, X_train)
    if args.cuda:
        vade = vade.cuda()

    X_train_all_np, pre_res = vade.pre_train(pre_epoch=500)
    build_gp(X_train_all_np, pre_res, args)








