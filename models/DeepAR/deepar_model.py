import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class Gaussian(nn.Module):

    def __init__(self, hidden_size, output_size):
        '''
        Gaussian Likelihood Supports Continuous Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(Gaussian, self).__init__()
        self.mu_layer = nn.Linear(hidden_size, output_size)
        self.sigma_layer = nn.Linear(hidden_size, output_size)

        # initialize weights
        # nn.init.xavier_uniform_(self.mu_layer.weight)
        # nn.init.xavier_uniform_(self.sigma_layer.weight)

    def forward(self, h):
        _, hidden_size = h.size()
        sigma_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-6
        sigma_t = sigma_t.squeeze(0)
        mu_t = self.mu_layer(h).squeeze(0)
        return mu_t, sigma_t


class NegativeBinomial(nn.Module):

    def __init__(self, input_size, output_size):
        '''
        Negative Binomial Supports Positive Count Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(NegativeBinomial, self).__init__()
        self.mu_layer = nn.Linear(input_size, output_size)
        self.sigma_layer = nn.Linear(input_size, output_size)

    def forward(self, h):
        _, hidden_size = h.size()
        alpha_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-6
        mu_t = torch.log(1 + torch.exp(self.mu_layer(h)))
        return mu_t, alpha_t


def gaussian_sample(mu, sigma):
    '''
    Gaussian Sample
    Args:
    ytrue (array like)
    mu (array like)
    sigma (array like): standard deviation

    gaussian maximum likelihood using log
        l_{G} (z|mu, sigma) = (2 * pi * sigma^2)^(-0.5) * exp(- (z - mu)^2 / (2 * sigma^2))
    '''
    # likelihood = (2 * np.pi * sigma ** 2) ** (-0.5) * \
    #         torch.exp((- (ytrue - mu) ** 2) / (2 * sigma ** 2))
    # return likelihood
    gaussian = torch.distributions.normal.Normal(mu, sigma)
    # ypred = gaussian.sample(mu.size())
    ypred = gaussian.sample()
    return ypred


def negative_binomial_sample(mu, alpha):
    '''
    Negative Binomial Sample
    Args:
    ytrue (array like)
    mu (array like)
    alpha (array like)

    maximuze log l_{nb} = log Gamma(z + 1/alpha) - log Gamma(z + 1) - log Gamma(1 / alpha)
                - 1 / alpha * log (1 + alpha * mu) + z * log (alpha * mu / (1 + alpha * mu))

    minimize loss = - log l_{nb}

    Note: torch.lgamma: log Gamma function
    '''
    var = mu + mu * mu * alpha
    ypred = mu + torch.randn(mu.size()) * torch.sqrt(var)
    return ypred


class DeepAR(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, lr=1e-3, likelihood="g"):
        super(DeepAR, self).__init__()

        # network
        self.input_embed = nn.Linear(1, embedding_size)
        self.encoder = nn.LSTM(embedding_size + input_size, hidden_size, \
                               num_layers, bias=True, batch_first=True)
        if likelihood == "g":
            self.likelihood_layer = Gaussian(hidden_size, 1)
        elif likelihood == "nb":
            self.likelihood_layer = NegativeBinomial(hidden_size, 1)
        self.likelihood = likelihood

    def forward(self, X, y, Xf):
        '''
        Args:
        X (array like): shape (num_time_series, seq_len, input_size)
        y (array like): shape (num_time_series, seq_len)
        Xf (array like): shape (num_time_series, horizon, input_size)
        Return:
        mu (array like): shape (batch_size, seq_len)
        sigma (array like): shape (batch_size, seq_len)
        '''
        if isinstance(X, type(np.empty(2))):
            X = torch.from_numpy(X).float()
            y = torch.from_numpy(y).float()
            Xf = torch.from_numpy(Xf).float()
        num_ts, seq_len, _ = X.size()
        _, output_horizon, num_features = Xf.size()
        ynext = None
        ypred = []
        mus = []
        sigmas = []
        h, c = None, None
        for s in range(seq_len + output_horizon):
            if s < seq_len:
                ynext = y[:, s].view(-1, 1)
                yembed = self.input_embed(ynext).view(num_ts, -1)
                x = X[:, s, :].view(num_ts, -1)
            else:
                yembed = self.input_embed(ynext).view(num_ts, -1)
                x = Xf[:, s - seq_len, :].view(num_ts, -1)
            x = torch.cat([x, yembed], dim=1)  # num_ts, num_features + embedding
            inp = x.unsqueeze(1)
            # print("inp shape:{}".format(inp.shape))
            if h is None and c is None:
                out, (h, c) = self.encoder(inp)  # h size (num_layers, num_ts, hidden_size)
            else:
                out, (h, c) = self.encoder(inp, (h, c))
            hs = h[-1, :, :]
            hs = F.relu(hs)
            mu, sigma = self.likelihood_layer(hs)
            mus.append(mu.view(-1, 1))
            sigmas.append(sigma.view(-1, 1))
            # print('ynext shape1:{}'.format(ynext.shape))
            if self.likelihood == "g":
                ynext = gaussian_sample(mu, sigma)
            elif self.likelihood == "nb":
                alpha_t = sigma
                mu_t = mu
                ynext = negative_binomial_sample(mu_t, alpha_t)
            # print('ynext shape2:{}'.format(ynext.shape))
            # if without true value, use prediction
            if s >= seq_len - 1 and s < output_horizon + seq_len - 1:
                ypred.append(ynext)
        ypred = torch.cat(ypred, dim=1).view(num_ts, -1)
        mu = torch.cat(mus, dim=1).view(num_ts, -1)
        sigma = torch.cat(sigmas, dim=1).view(num_ts, -1)
        return ypred, mu, sigma