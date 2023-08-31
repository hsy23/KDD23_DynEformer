import argparse
from dataloader import *
from model_vade import VaDE
import numpy as np
from build_global_pool import build_global_pool
import pickle as pkl


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='VaDE')
    parse.add_argument('--batch_size', type=int, default=40000)
    parse.add_argument('--dataX_dir', type=str, default='../../../data/ECW_08.npy')
    parse.add_argument('--nClusters', type=int, default=150)

    parse.add_argument('--series_len', type=int, default=48)
    parse.add_argument('--step', type=int, default=12)

    parse.add_argument('--hid_dim', type=int, default=10)
    parse.add_argument('--cuda', type=bool, default=True)

    parse.add_argument('--infer_test', type=bool, default=False)
    parse.add_argument('--infer_data_path', type=str, default='../../../data/ECW_switch.pkl')

    args=parse.parse_args()

    X = np.load(open(args.dataX_dir, 'rb'), allow_pickle=True)
    X_train = X[:, :24*25, :]

    vade = VaDE(args, X_train)
    if args.cuda:
        vade = vade.cuda()

    decomp_X, clusters = vade.pre_train(pre_epoch=500)
    global_pool = build_global_pool(decomp_X, clusters, args.nClusters, args)

    pkl.dump(global_pool, open('global_pool_c{}_s{}_s{}.pkl'.format(args.nClusters, args.series_len, args.step), 'wb'))

    if args.infer_test:
        data_path = args.infer_data_path
        X = pickle.load(open(data_path, 'rb'))
        vade.predict(X, 'mac')
        # vade.predict(x_app, 'app')
        # build_global_pool(551, args)







