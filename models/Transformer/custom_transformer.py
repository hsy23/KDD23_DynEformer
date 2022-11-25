import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *


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

        a = self.fc1(self.dropout(F.relu(self.fc2(x))))
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

        a = self.fc1(self.dropout(F.relu(self.fc2(x))))

        x = self.norm3(x + self.dropout(a))
        return x


class Transformer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, dec_seq_len, out_seq_len, n_decoder_layers=1, n_encoder_layers=1,
                 n_heads=1, dropout=0.1):
        super(Transformer, self).__init__()  # 继承父类torch.nn.Module的初始化方法
        self.dec_seq_len = dec_seq_len

        # Initiate encoder and Decoder layers
        self.encs = nn.ModuleList()
        for i in range(n_encoder_layers):
            self.encs.append(EncoderLayer(dim_val, dim_attn, dropout, n_heads))

        self.decs = nn.ModuleList()
        for i in range(n_decoder_layers):
            self.decs.append(DecoderLayer(dim_val, dim_attn, dropout, n_heads))

        self.pos = PositionalEncoding(dim_val)

        # Dense layers for managing network inputs and outputs
        self.enc_input_fc = nn.Linear(input_size, dim_val)
        self.dec_input_fc = nn.Linear(input_size, dim_val)
        self.out_fc = nn.Linear(dec_seq_len * dim_val, out_seq_len)

    def forward(self, x):
        # x = x.transpose(0, 1)  # 交换0/1维度
        # encoder
        e = self.encs[0](self.pos(self.enc_input_fc(x)))
        for enc in self.encs[1:]:
            e = enc(e)

        # decoder
        d = self.decs[0](self.dec_input_fc(x[:, -self.dec_seq_len:]), e)
        for dec in self.decs[1:]:
            d = dec(d, e)

        # output
        # d = d.transpose(0, 1)  # 交换0/1维度
        x = self.out_fc(d.flatten(start_dim=1))

        return x


