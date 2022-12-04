from torch import nn
import argparse
import math
from custom_transformer_token import Transformer

import sys
sys.path.append('../')

from data_process_utils import *
from global_utils import *
from tqdm import tqdm
from torch.optim import Adam
import torch
import time
from torch.utils.data import DataLoader


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x


class TransformerTS(nn.Module):
    def __init__(self,
                 input_dim,
                 dec_seq_len,
                 out_seq_len,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 custom_encoder=None,
                 custom_decoder=None):
        r"""A transformer model. User is able to modify the attributes as needed. The architecture
        is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
        Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
        Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
        Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
        model with corresponding parameters.

        Args:
            input_dim: dimision of imput series
            d_model: the number of expected features in the encoder/decoder inputs (default=512).
            nhead: the number of heads in the multiheadattention models (default=8).
            num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
            num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
            dim_feedforward: the dimension of the feedforward network model (default=2048).
            dropout: the dropout value (default=0.1).
            activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
            custom_encoder: custom encoder (default=None).
            custom_decoder: custom decoder (default=None).

        Examples::
            # >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
            # >>> src = torch.rand((10, 32, 512)) (time length, N, feature dim)
            # >>> tgt = torch.rand((20, 32, 512))
            # >>> out = transformer_model(src, tgt)

        Note: A full example to apply nn.Transformer module for the word language model is available in
        https://github.com/pytorch/examples/tree/master/word_language_model
        """
        super(TransformerTS, self).__init__()
        self.transform = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            custom_encoder=custom_encoder,
            custom_decoder=custom_decoder,
        )
        self.pos = PositionalEncoding(d_model)
        self.enc_input_fc = nn.Linear(input_dim, d_model)
        self.dec_input_fc = nn.Linear(input_dim, d_model)
        self.out_fc = nn.Linear(dec_seq_len * d_model, out_seq_len)
        self.dec_seq_len = dec_seq_len

    def forward(self, x):
        x = x.transpose(0, 1)  # 交换0/1维度
        # embedding
        embed_encoder_input = self.pos(self.enc_input_fc(x))
        embed_decoder_input = self.dec_input_fc(x[-self.dec_seq_len:, :])
        # transform
        x = self.transform(embed_encoder_input, embed_decoder_input)

        # output
        x = x.transpose(0, 1)
        x = self.out_fc(x.flatten(start_dim=1))
        return x


def batch_generator(X, y, num_obs_to_train, seq_len, batch_size):
    '''
    Args:
    X (array like): shape (num_samples, num_features, num_periods)
    y (array like): shape (num_samples, num_periods)
    num_obs_to_train (int):
    seq_len (int): sequence/encoder/decoder length
    batch_size (int)
    '''
    num_ts, num_periods, _ = X.shape
    if num_ts < batch_size:  # 序列数
        batch_size = num_ts
    t = random.choice(range(num_obs_to_train, num_periods - seq_len))  # 一次输入的截断数据下标t前为预测前输入 t后为被预测序列输入
    batch = random.sample(range(num_ts), batch_size)
    X_train_batch = X[batch, t - num_obs_to_train:t, :]
    y_train_batch = y[batch, t:t + seq_len]
    return X_train_batch, y_train_batch


def batch_generator_all(X, Y, num_obs_to_train, seq_len):
    '''
        Args:
        X (array like): shape (num_samples, num_features, num_periods)
        y (array like): shape (num_samples, num_periods)
        num_obs_to_train (int):
        seq_len (int): sequence/encoder/decoder length
        batch_size (int)
        '''
    num_ts, num_periods, n_feats = X.shape
    X_train_all = []
    Y_train_all = []

    for i in range(num_ts):
        for j in range(num_obs_to_train, num_periods - seq_len, 12):
            X_train_all.append(X[i, j-num_obs_to_train:j, :])
            Y_train_all.append(Y[i, j:j+seq_len])

    X_train_all = np.asarray(X_train_all).reshape(-1, num_obs_to_train, n_feats)
    Y_train_all = np.asarray(Y_train_all).reshape(-1, seq_len)
    return X_train_all, Y_train_all


def batch_generator_padding(X, Y, num_obs_to_train, seq_len):
    '''
        Args:
        X (array like): shape (num_samples, num_features, num_periods)
        y (array like): shape (num_samples, num_periods)
        num_obs_to_train (int):
        seq_len (int): sequence/encoder/decoder length
        batch_size (int)
        '''
    num_ts, num_periods, n_feats = X.shape
    X_all = []
    Y_all = []

    for i in range(num_ts):
        for j in range(num_obs_to_train, num_periods - seq_len, 12):
            X_all.append(X[i, j-num_obs_to_train:j+seq_len, :])
            Y_all.append(Y[i, j:j+seq_len])

    X_all = np.asarray(X_all).reshape(-1, num_obs_to_train+seq_len, n_feats)
    Y_all = np.asarray(Y_all).reshape(-1, seq_len)
    return X_all, Y_all


def train(X, y, args):
    '''
    Args:
    - X (array like): shape (num_samples, num_features, num_periods)
    - y (array like): shape (num_samples, num_periods)
    - epoches (int): number of epoches to run
    - step_per_epoch (int): steps per epoch to run
    - seq_len (int): output horizon
    - likelihood (str): what type of likelihood to use, default is gaussian
    - num_skus_to_show (int): how many skus to show in test phase
    - num_results_to_sample (int): how many samples in test phase as prediction
    '''

    device = torch.device('cuda:0')

    num_ts, num_periods, num_features = X.shape
    # model = TransformerTS(num_features,
    #                       args.dec_seq_len,
    #                       args.out_seq_len,
    #                       d_model=args.d_model,
    #                       nhead=args.nhead,
    #                       num_encoder_layers=args.n_encoder_layers,
    #                       num_decoder_layers=args.n_decoder_layers,
    #                       dim_feedforward=args.dim_feedforward,
    #                       dropout=args.dropout,
    #                       activation=args.activation,
    #                       custom_encoder=None,
    #                       custom_decoder=None)
    model = Transformer(args.d_model, args.d_model, num_features, args.enc_seq_len, args.dec_seq_len, args.out_seq_len,
                        n_encoder_layers=args.n_encoder_layers, n_decoder_layers=args.n_decoder_layers,
                        n_heads=args.nhead, dropout=args.dropout)

    # model = pickle.load(open("deepar.pkl", 'rb'))
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    random.seed(2)
    # select sku with most top n quantities
    Xtr, ytr, Xte, yte = train_test_split(X, y, train_ratio=0.8)

    losses = []
    test_mse = []
    test_mae = []
    mse = nn.MSELoss().to(device)
    cnt = 0

    yscaler = None
    if args.standard_scaler:
        yscaler = StandardScaler()
    elif args.log_scaler:
        yscaler = LogScaler()
    elif args.mean_scaler:
        yscaler = MeanScaler()
    elif args.minmax_scaler:
        yscaler = MinMaxScaler()
    if yscaler is not None:
        ytr = yscaler.fit_transform(ytr)

    
    pre_seq_len = args.out_seq_len
    num_obs_to_train = args.enc_seq_len
    X_train_all, Y_train_all = batch_generator_padding(Xtr, ytr, num_obs_to_train, pre_seq_len)
    X_test_all, Y_test_all = batch_generator_padding(Xte, yte, num_obs_to_train, pre_seq_len)
    
    # training
    for epoch in tqdm(range(args.num_epoches)):
        # print("Epoch {} starts...".format(epoch))
        for step in range(int(len(X_train_all)/args.batch_size)):
            # Xtrain, ytrain = batch_generator(Xtr, ytr, num_obs_to_train, pre_seq_len, args.batch_size)
            Xtrain, ytrain = X_train_all[step*args.batch_size:(step+1)*args.batch_size, :, :], \
                             Y_train_all[step*args.batch_size:(step+1)*args.batch_size, :]
            Xtrain_tensor = torch.from_numpy(Xtrain.astype(float)).float().to(device)
            ytrain_tensor = torch.from_numpy(ytrain.astype(float)).float().to(device)

            ypred = model(Xtrain_tensor)
            loss = mse(ypred, ytrain_tensor)
            if (epoch % 10 == 0 and step == 0):
                print('The Train MSE Loss {}'.format(loss.item()))
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1

        with torch.no_grad():
            test_epoch_loss = []
            test_epoch_mse = []
            test_epoch_mae = []

            for step in range(int(len(X_test_all) / args.batch_size)):
                Xtest, ytest = X_test_all[step * args.batch_size:(step + 1) * args.batch_size, :, :], \
                               Y_test_all[step * args.batch_size:(step + 1) * args.batch_size, :]

                Xtest_tensor = torch.from_numpy(Xtest.astype(float)).float().to(device)
                ytest_tensor = torch.from_numpy(ytest.astype(float)).float().to(device)
                
                yPred_test = model(Xtest_tensor)

                test_epoch_loss.append(mse(yPred_test, ytest_tensor).item())
                yPred_test = yPred_test.cpu().numpy()

                if yscaler is not None:
                    yPred_test = yscaler.inverse_transform(yPred_test)

                test_epoch_mse.append(((ytest.reshape(-1) - yPred_test.reshape(-1)) ** 2).mean())
                test_epoch_mae.append(np.abs(ytest.reshape(-1) - yPred_test.reshape(-1)).mean())

            if epoch % 10 == 0:
                print('The Test MSE Loss is {}'.format(np.average(test_epoch_loss)))
                print('The Mean Squared Error of forecasts is {} (raw)'.format(np.average(test_epoch_mse)))
                print('The Mean Absolute Error of forecasts is {} (raw)'.format(np.average(test_epoch_mae)))


    model = model.cpu()
    if args.save_model:
        pickle.dump(model, open("transformer_{}.pkl".format(time.strftime("%m%d%H%M", time.localtime())), 'wb'))
    if args.load_model:
        model = pickle.load(open("transformer_11092318.pkl", 'rb'))

    # test
    # X_test = Xte[:, -pre_seq_len - num_obs_to_train:, :].reshape((num_ts, -1, num_features))
    # y_test = yte[:, -pre_seq_len:].reshape((num_ts, -1))

    # test_sample = np.random.choice(range(len(X_test)), 100, replace=False)
    # X_test = X_test[test_sample]
    # y_test = y_test[test_sample]

    # if yscaler is not None:
    #     y_test = yscaler.transform(y_test)
    

    if args.show_plot:
        draw_id = np.random.randint(0, 100, 1)
        plt.figure(figsize=(20, 5))

        plt.plot(range(args.enc_seq_len + y_test[draw_id].shape[1]),
                 (X_test[draw_id, :, 0].squeeze(0)), label="true")
        plt.plot(range(args.enc_seq_len, args.enc_seq_len + y_pred_t[draw_id].shape[1]), y_pred_t[draw_id].squeeze(0),
                 label="forecast", color='r')
        # plt.plot(range(args.num_obs_to_train, args.num_obs_to_train + y_test[draw_id].shape[1]), ,
        #          label="true_future")

        plt.title('Prediction Value')
        plt.legend(loc="upper left")
        ymin, ymax = plt.ylim()
        plt.ylim(ymin, ymax)
        plt.xlabel("Periods")
        plt.ylabel("Y")
        plt.show()
    return losses
# if __name__ == "__main__":
#     import torch
#     src = torch.rand(
#         (32, 10, 512))  # 32:batch size 10:time length 512: feature dimension
#     input_dim = src.shape[-1]
#     lat_num = src.shape[1]
#     dec_seq_len = min(lat_num, 10)  # decoder部分使用的长度
#     d_model = 512  # 嵌入维度
#     nhead = 8
#     num_encoder_layers = 6
#     num_decoder_layers = 6
#     dim_feedforward = 2048
#     dropout = 0.1
#     activation = 'relu'
#     out_seq_len = 10
#
#     transformer_model = TransformerTS(input_dim,
#                                       dec_seq_len,
#                                       out_seq_len,
#                                       d_model=d_model,
#                                       nhead=nhead,
#                                       num_encoder_layers=num_encoder_layers,
#                                       num_decoder_layers=num_decoder_layers,
#                                       dim_feedforward=dim_feedforward,
#                                       dropout=dropout,
#                                       activation=activation,
#                                       custom_encoder=None,
#                                       custom_decoder=None)
#
#     out = transformer_model(src)
#     print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard_scaler", "-ss", action="store_true")
    parser.add_argument("--log_scaler", "-ls", action="store_true")
    parser.add_argument("--mean_scaler", "-ms", action="store_true")
    parser.add_argument("--minmax_scaler", "-mm", action="store_true", default=True)

    parser.add_argument("--num_epoches", "-e", type=int, default=200)
    parser.add_argument("--step_per_epoch", "-spe", type=int, default=3)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", "-b", type=int, default=256)

    parser.add_argument("--n_encoder_layers", "-nel", type=int, default=2)
    parser.add_argument("--n_decoder_layers", "-ndl", type=int, default=1)
    parser.add_argument("--d_model", "-dm", type=int, default=128)  # 嵌入维度
    parser.add_argument("--nhead", "-nh", type=int, default=8)  # 注意力头数量
    parser.add_argument("--dim_feedforward", "-hs", type=int, default=256)
    parser.add_argument("--dec_seq_len", "-dl", type=int, default=12)  # decoder用到的输入长度
    parser.add_argument("--out_seq_len", "-ol", type=int, default=24)  # 预测长度
    parser.add_argument("--enc_seq_len", "-not", type=int, default=24*2)  # 输入训练长度
    parser.add_argument("-dropout", type=float, default=0.1)
    parser.add_argument("-activation", type=str, default='relu')

    parser.add_argument("--run_test", "-rt", action="store_true", default=True)
    parser.add_argument("--save_model", "-sm", type=bool, default=False)
    parser.add_argument("--load_model", "-lm", type=bool, default=False)
    parser.add_argument("--show_plot", "-sp", type=bool, default=False)

    parser.add_argument("--day_periods", "-dp", type=int, default=288)
    parser.add_argument("--num_periods", "-np", type=int, default=24)
    parser.add_argument("--num_days", "-ds", type=int, default=30)

    args = parser.parse_args()

    if args.run_test:
        # data_path = get_data_path("merged_20220801-20220830_bd.pkl")
        # # mac_attr_path = get_data_path("20220801-20220830_mac_attr.pkl")
        # # mac_pro_path = get_data_path("20220801-20220830_mac_pro.pkl")
        # # demand_area_path = get_data_path("20220830_demand_area.pkl")
        #
        # data = pd.read_pickle(open(data_path, 'rb'))
        # # mac_attr = pd.read_pickle(open(mac_attr_path, 'rb'))
        # # mac_pro = pd.read_pickle(open(mac_pro_path, 'rb'))
        # # demand_area = pd.read_pickle(open(demand_area_path, 'rb'))
        # # data = app_filter(data)
        # # data = data.sort_values(by=['machine_id', 'time_id'])
        #
        # # data = dt_2_dayl(data)
        # # data['bw_upload'] = data['bw_upload'] / 1024 / 1024 / 1024  # GB
        # # data = data.loc[(data["date"].dt.date >= date(2022, 8, 25)) & (data["date"].dt.date <= date(2014, 3, 1))]
        # # add_feats = data_pro_merge(mac_attr)
        # X_all = []
        # y_all = []
        #
        # add_features = ["hour", "day", "week"]
        # num_feats = len(add_features)+1
        # # 保留满足完整时序的数据
        # # tmp = data.groupby(['machine_id', 'name']).size().reset_index()
        # # tmp = data.groupby('machine_id').size().reset_index()
        # # tmp2 = tmp[tmp[0] == num_periods]
        # # tmp3 = pd.merge(data, tmp2, on=['machine_id', 'name']).iloc[:, :-1]
        # # tmp3 = pd.merge(data, tmp2, on=['machine_id']).iloc[:, :-1]
        #
        # for i in tqdm(range(0, len(data), args.day_periods*30)):
        #     s = data.iloc[i:i + args.day_periods*30, :]
        #     if len(s['machine_id'].unique()) != 1:
        #         print("发生切任务：", s['machine_id'])
        #     if len(s['machine_id'].unique()) != 1:
        #         raise ValueError("出现新mac")
        #     X = []
        #     y = []
        #     for j in range(0, len(s), 12):
        #         s_hour = s.iloc[j:j + 12]
        #         workload = np.max(s_hour['bw_upload'])
        #         X.append(workload)
        #         X.extend(s_hour[add_features].iloc[0].values)
        #         y.append(workload)
        #
        #     X = np.asarray(X).reshape((args.num_periods*30, num_feats))  # num_series
        #     y = np.asarray(y).reshape((args.num_periods*30))
        #
        #     X_all.append(X)
        #     y_all.append(y)
        #
        # X_all = np.asarray(X_all).reshape((-1, args.num_periods*30, num_feats))
        # y_all = np.asarray(y_all).reshape((-1, args.num_periods*30))
        X_all = pickle.load(open(get_data_path("X_all_0801_0830.pkl"), 'rb'))
        y_all = pickle.load(open(get_data_path("y_all_0801_0830.pkl"), 'rb'))
        losses = train(X_all, y_all, args)
        # pickle.dump(losses, open("losses_{}.pkl".format(time.strftime("%m%d%H%M", time.localtime())), 'wb'))
        if args.show_plot:
            plt.plot(range(len(losses)), losses, "k-")
            plt.xlabel("Period")
            plt.ylabel("Loss")
            plt.show()