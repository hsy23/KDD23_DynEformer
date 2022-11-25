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
from dataloader import get_pworkload, get_pworkload_all
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
    def __init__(self, args, X, Y):
        super(VaDE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.pi_ = nn.Parameter(torch.FloatTensor(args.nClusters,).fill_(1)/args.nClusters, requires_grad=True)
        self.mu_c = nn.Parameter(torch.FloatTensor(args.nClusters, args.hid_dim).fill_(0), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.FloatTensor(args.nClusters, args.hid_dim).fill_(0), requires_grad=True)

        self.args=args
        self.X = X
        self.Y = Y

    def pre_train(self, pre_epoch=10):
        if not os.path.exists('./pretrain_model.pk'):
            Loss = nn.MSELoss()
            opti = Adam(itertools.chain(self.encoder.parameters(),self.decoder.parameters()), lr=1e-3)

            print('Pretraining......')
            epoch_bar = tqdm(range(pre_epoch))
            x_all, _ = get_pworkload_all(self.X, self.Y, std='minmax')
            for i in epoch_bar:
                L = 0
                # for step in range(self.args.step_per_epoch):
                if self.args.cuda:
                    x = x_all.cuda()

                z, _ = self.encoder(x)
                x_ = self.decoder(z)
                # if i % 100 == 0:
                #     tmp_x = x.detach().cpu().numpy()
                #     tmpx_ = x_.detach().cpu().numpy()
                #     fig = plt.figure()
                #     m_id = np.random.randint(0, len(tmp_x), 1)
                #     plt.plot(tmp_x[m_id[0]], label='origin')
                #     plt.plot(tmpx_[m_id[0]], label='pre')
                #     plt.legend()
                #     plt.show()

                loss = Loss(x, x_)
                L = loss.detach().cpu().numpy()
                if i % 100 == 0:
                    epoch_bar.write('L2={:.4f}'.format(L))

                opti.zero_grad()
                loss.backward()
                opti.step()

            self.encoder.log_sigma2_l.load_state_dict(self.encoder.mu_l.state_dict())

            Z = []
            Y = []
            with torch.no_grad():
                for x in x_all:
                    if self.args.cuda:
                        x = x.cuda()

                    z1, z2 = self.encoder(x)
                    assert F.mse_loss(z1, z2) == 0
                    Z.append(z1)
                    # Y.append(y)

            Z = torch.cat(Z, 0).detach().cpu().numpy()  # 将列表合成一个tensor
            # Y = torch.cat(Y, 0).detach().numpy()

            # aic_l = []
            # bic_l = []
            # for n_c in tqdm(range(1, 800, 50)):
            #     gmm = GaussianMixture(n_components=self.args.nClusters, covariance_type='diag')
            #     gmm_model = gmm.fit(Z.reshape(-1, 10))
            #     # print('aic:', gmm_model.aic(Z.reshape(-1, 10)))
            #     aic_l.append(gmm_model.aic(Z.reshape(-1, 10)))
            #     bic_l.append(gmm_model.bic(Z.reshape(-1, 10)))

            # pickle.dump(aic_l, open('./trend_aic_l.pkl', 'wb'))
            # pickle.dump(bic_l, open('./trend_bic_l.pkl', 'wb'))
            # plt.plot(aic_l, label='aic')
            # plt.plot(bic_l, label='bic')
            # plt.xlabel('n_clusters')
            # plt.ylabel('aic/bic')
            # plt.xticks(range(len(aic_l)), labels=range(1, 800, 50))
            # plt.title('trend clusters valuation')
            # plt.show()
            # print('Acc={:.4f}%'.format(cluster_acc(pre, Y)[0] * 100))

            gmm = GaussianMixture(n_components=self.args.nClusters, covariance_type='diag')
            gmm_model = gmm.fit(Z.reshape(-1, 10))
            pre_res = gmm_model.predict(Z.reshape(-1, 10))
            self.pi_.data = torch.from_numpy(gmm_model.weights_).cuda().float()
            self.mu_c.data = torch.from_numpy(gmm_model.means_).cuda().float()
            self.log_sigma2_c.data = torch.log(torch.from_numpy(gmm_model.covariances_).cuda().float())
            torch.save(self.state_dict(), './pretrain_model_seasonal_nc150.pk')
            pickle.dump(pre_res, open('seasonal_cluster_res', 'wb'))
        else:
            self.load_state_dict(torch.load('./pretrain_model_trend_nc150.pk'))

    def predict(self, x):
        z_mu, z_sigma2_log = self.encoder(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z, mu_c, log_sigma2_c))

        yita = yita_c.detach().cpu().numpy()
        return np.argmax(yita, axis=1)

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


