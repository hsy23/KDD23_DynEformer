import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
import itertools
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np
import os
from dataloader import get_pworkload
import matplotlib.pyplot as plt


def cluster_acc(Y_pred, Y):

    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w


def block(in_c,out_c):
    layers=[
        nn.Linear(in_c,out_c),
        nn.ReLU(True)
    ]
    return layers


class Encoder(nn.Module):
    def __init__(self,input_dim=48,inter_dims=[500, 500, 2000], hid_dim=10):
        super(Encoder,self).__init__()

        self.encoder=nn.Sequential(
            *block(input_dim,inter_dims[0]),
            *block(inter_dims[0],inter_dims[1]),
            *block(inter_dims[1],inter_dims[2]),
        )

        self.mu_l = nn.Linear(inter_dims[-1], hid_dim)
        self.log_sigma2_l = nn.Linear(inter_dims[-1], hid_dim)

    def forward(self, x):
        e = self.encoder(x)

        mu = self.mu_l(e)
        log_sigma2 = self.log_sigma2_l(e)

        return mu, log_sigma2


class Decoder(nn.Module):
    def __init__(self,input_dim=48,inter_dims=[500, 500, 2000],hid_dim=10):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            *block(hid_dim, inter_dims[-1]),
            *block(inter_dims[-1], inter_dims[-2]),
            *block(inter_dims[-2], inter_dims[-3]),
            nn.Linear(inter_dims[-3], input_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        x_pro=self.decoder(z)

        return x_pro


class VaDE(nn.Module):
    def __init__(self, args, X):
        super(VaDE, self).__init__()
        self.encoder = Encoder(input_dim=args.series_len)
        self.decoder = Decoder(input_dim=args.series_len)

        self.pi_ = nn.Parameter(torch.FloatTensor(args.nClusters,).fill_(1)/args.nClusters, requires_grad=True)
        self.mu_c = nn.Parameter(torch.FloatTensor(args.nClusters, args.hid_dim).fill_(0), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.FloatTensor(args.nClusters, args.hid_dim).fill_(0), requires_grad=True)

        self.args=args
        self.X = X
        self.Y = None

    def pre_train(self, pre_epoch=10):
        Loss = nn.MSELoss()
        opti = Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=1e-3)

        print('Pretraining......')
        epoch_bar = tqdm(range(pre_epoch))
        x_all_loader, X_train_all, X_train_all_np = get_pworkload(self.X, self.Y, std='minmax', series_len=self.args.series_len, step=self.args.step, batch_size=self.args.batch_size)
        for i in epoch_bar:
            L = 0
            if self.args.cuda:
                x = X_train_all.cuda()

            z, _ = self.encoder(x)
            x_ = self.decoder(z)

            loss = Loss(x, x_)
            L = loss.detach().cpu().numpy()

            opti.zero_grad()
            loss.backward()
            opti.step()

            epoch_bar.write('L2={:.4f}'.format(L))

        self.encoder.log_sigma2_l.load_state_dict(self.encoder.mu_l.state_dict())

        Z = []
        Y = []
        with torch.no_grad():
            if self.args.cuda:
                x = X_train_all.cuda()

            z1, z2 = self.encoder(x)
            assert F.mse_loss(z1, z2) == 0
            Z.append(z1)

        Z = torch.cat(Z, 0).detach().cpu().numpy()  # 将列表合成一个tensor
        best_n = self.args.nClusters

        gmm = GaussianMixture(n_components=best_n, covariance_type='diag')
        gmm_model = gmm.fit(Z.reshape(-1, 10))
        best_gmm = gmm_model
        pre_res = best_gmm.predict(Z.reshape(-1, 10))
        self.pi_.data = torch.from_numpy(best_gmm.weights_).cuda().float()
        self.mu_c.data = torch.from_numpy(best_gmm.means_).cuda().float()
        self.log_sigma2_c.data = torch.log(torch.from_numpy(best_gmm.covariances_).cuda().float())
        return X_train_all_np, pre_res

    def predict(self, x, type):
        _, X_test_all, X_raw_all, y_test_all, y_raw_all = get_pworkload(x, None, wtype='test', std='minmax',
                                                               series_len=self.args.series_len, step=self.args.step,
                                                               batch_size=self.args.batch_size)
        z_mu, z_sigma2_log = self.encoder(X_test_all.cuda())
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z, mu_c, log_sigma2_c))
        yita = yita_c.detach().cpu().numpy()
        return [X_raw_all, y_raw_all, np.argmax(yita, axis=1)]

    def ELBO_Loss(self,x,L=1):
        det=1e-10

        L_rec=0

        z_mu, z_sigma2_log = self.encoder(x)
        for l in range(L):

            z=torch.randn_like(z_mu)*torch.exp(z_sigma2_log/2)+z_mu

            x_pro=self.decoder(z)

            L_rec+=F.binary_cross_entropy(x_pro,x)

        L_rec/=L

        Loss=L_rec*x.size(1)

        pi=self.pi_
        log_sigma2_c=self.log_sigma2_c
        mu_c=self.mu_c

        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        yita_c=torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))+det

        yita_c=yita_c/(yita_c.sum(1).view(-1,1))#batch_size*Clusters

        Loss+=0.5*torch.mean(torch.sum(yita_c*torch.sum(log_sigma2_c.unsqueeze(0)+
                                                torch.exp(z_sigma2_log.unsqueeze(1)-log_sigma2_c.unsqueeze(0))+
                                                (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/torch.exp(log_sigma2_c.unsqueeze(0)),2),1))

        Loss-=torch.mean(torch.sum(yita_c*torch.log(pi.unsqueeze(0)/(yita_c)),1))+0.5*torch.mean(torch.sum(1+z_sigma2_log,1))


        return Loss

    def gaussian_pdfs_log(self, x, mus, log_sigma2s):
        G = []
        for c in range(self.args.nClusters):
            G.append(self.gaussian_pdf_log(x, mus[c:c+1,:], log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G, 1)

    @staticmethod
    def gaussian_pdf_log(x,mu,log_sigma2):
        return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))


