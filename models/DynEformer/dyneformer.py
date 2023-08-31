import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *


class GpLayer(nn.Module):
    def __init__(self, dim_val, time_feature_len, enc_seq_len):
        super().__init__()
        self.dim_val = dim_val
        half = int(dim_val/2)
        self.fuse_src = nn.Linear(
            in_features=enc_seq_len*half,
            out_features=time_feature_len
            )
        self.recover = nn.Linear(
            in_features=time_feature_len,
            out_features=half
        )

    def forward(self, src, timeFeature):
        half = int(self.dim_val/2)
        tmp = src[:, :, half:].flatten(1)
        key = torch.softmax(self.fuse_src(tmp), 1)
        fuse = key * timeFeature.unsqueeze(1)
        out = torch.concat((src[:, :, :half], self.recover(fuse).transpose(0, 1)), dim=-1)
        return key, out


class SaLayer(nn.Module):
    def __init__(self, dim_val, dim_w, dim_static, dropout):
        super().__init__()
        self.dim_val = dim_val
        half = int(dim_val/2)
        self.fuse_src = nn.Linear(
            in_features=dim_val,
            out_features=dim_w
            )
        self.static_embedding = nn.Linear(dim_static, dim_w)
        self.norm = nn.LayerNorm(dim_val)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, static_context):
        q = self.fuse_src(src)
        v = self.static_embedding(static_context)
        key = torch.softmax(q, -1)
        fuse = key * v.unsqueeze(1)
        return self.norm(src + self.dropout(fuse))


class EncoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, dropout, n_heads=1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)

        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        a = self.attn(x)
        x = self.norm1(x + self.dropout(a))

        a = self.fc2(self.dropout(F.relu(self.fc1(x))))
        x = self.norm2(x + self.dropout(a))

        return x


class DecoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, dropout, n_heads=1):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.attn2 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)

        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
        self.norm3 = nn.LayerNorm(dim_val)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc):
        a = self.attn1(x)
        x = self.norm1(x + self.dropout(a))

        a = self.attn2(x, kv=enc)
        x = self.norm2(x + self.dropout(a))

        a = self.fc2(self.dropout(F.relu(self.fc1(x))))

        x = self.norm3(x + self.dropout(a))
        return x


class DynEformer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, enc_seq_len, dec_seq_len, out_seq_len,
                 n_decoder_layers=1, n_encoder_layers=1, n_heads=1, dropout=0.5,
                 time_feature_len=150, dim_static=12):
        super(DynEformer, self).__init__()  # 继承父类torch.nn.Module的初始化方法
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.out_seq_len = out_seq_len

        self.static_layer = SaLayer(dim_val, dim_val, dim_static, dropout)

        # Initiate encoder and Decoder layers
        self.encs = nn.ModuleList()
        for i in range(n_encoder_layers):
            self.encs.append(EncoderLayer(dim_val, dim_attn, dropout, n_heads))

        self.add_feature_layer = GpLayer(dim_val, time_feature_len, enc_seq_len)

        self.decs = nn.ModuleList()
        for i in range(n_decoder_layers):
            self.decs.append(DecoderLayer(dim_val, dim_attn, dropout, n_heads))

        self.pos = PositionalEncoding(dim_val)

        # Dense layers for managing network inputs and outputs
        self.enc_input_fc = nn.Linear(input_size, dim_val)
        self.dec_input_fc = nn.Linear(input_size, dim_val)
        self.out_fc = nn.Linear(dim_val, 1)

    def forward(self, x, static_context, time_feature):
        # x = x.transpose(0, 1)
        # encoder

        new_x = torch.clone(x)
        enc_input = new_x[:, :self.enc_seq_len, :]
        dec_input = new_x[:, -self.dec_seq_len-self.out_seq_len:, :]

        e = self.pos(self.enc_input_fc(enc_input))
        e = self.static_layer(e, static_context)
        e = self.encs[0](e)
        _, e = self.add_feature_layer(e, time_feature)  # key dim = [batch_size, time_feature_len]
        for enc in self.encs[1:]:
            e = enc(e)
            key, e = self.add_feature_layer(e, time_feature)  # key dim = [batch_size, time_feature_len]

        # key, e = self.add_feature_layer(e, time_feature)  # key dim = [batch_size, time_feature_len]
        global_padding = torch.matmul(key, time_feature.transpose(0, 1))  # dim = [batch_size, enc_seq_len]
        dec_input[:, -self.out_seq_len:, 0] = global_padding[:, -self.out_seq_len:]  # padding

        # decoder
        d = self.dec_input_fc(dec_input)
        d = self.static_layer(d, static_context)
        d = self.decs[0](d, e)
        for dec in self.decs[1:]:
            d = dec(d, e)

        # output
        x = self.out_fc(d)[:, -self.out_seq_len:, :].squeeze(-1)

        return x


