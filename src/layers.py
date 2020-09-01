# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/8/26 14:05
# software: PyCharm

import torch.nn as nn
import torch
import numpy as np


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.9, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class FCLayerSoftmax(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.9, use_activation=True):
        super(FCLayerSoftmax, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        x1 = self.linear(x)
        y = self.softmax(x1)
        return y


class FCLayerSigmoid(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.9, use_activation=True):
        super(FCLayerSigmoid, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        x1 = self.linear(x)
        y = self.sigmoid(x1)
        return y


class BertPool(nn.Module):
    def __init__(self, config):
        super(BertPool, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SelfAttention(nn.Module):

    def __init__(self, sentence_num=0, key_size=0, hidden_size=0, attn_dropout=0.1):

        super(SelfAttention, self).__init__()
        self.linear_k = nn.Linear(hidden_size, key_size, bias=False)
        self.linear_q = nn.Linear(hidden_size, key_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dim_k = np.power(key_size, 0.5)
        self.softmax = nn.Softmax(1)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x, mask=None):
        """
        :param x:  [batch_size, max_seq_len, embedding_size]
        :param mask:
        :return:   [batch_size, embedding_size]
        """
        k = self.linear_k(x)
        q = self.linear_q(x)
        v = self.linear_v(x)
        # f = self.softmax(q.matmul(k.t()) / self.dim_k)
        attn = torch.bmm(q, k.transpose(1, 2)) / self.dim_k
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class KongJianTrans(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(KongJianTrans, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):

        return self.linear(x)
