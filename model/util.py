import torch
import numpy as np
import torch.nn as nn
import copy
import math
from torch.nn import functional as F


def Linear(inputdim, outputdim, bias=True):
    linear = nn.Linear(inputdim, outputdim, bias)
    return linear


def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_num, d_model, dropout=0.0):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % head_num == 0
        self.d_k = d_model // head_num
        self.head = head_num
        self.linears = clone(Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # print(scores.size(), mask.size())
        # when searching, target mask is not needed
        if mask is not None:
            # b 1 t -> b 1 1 t -> b head t t
            # print(scores.size())
            # print(mask)
            mask = mask.unsqueeze(1).expand_as(scores)
            # print(mask.size())
            scores.masked_fill_(mask == 0, -1e9)
            # print(scores)
        p_att = F.softmax(scores, -1)
        # print(p_att)
        # exit()
        if self.dropout:
            p_att = self.dropout(p_att)
        return torch.matmul(p_att, v)

    def forward(self, query, key, value, mask=None):
        # q k v : B T H
        nbatches = query.size(0)
        # b head t dim
        query, key, value = [l(x).view(nbatches, -1, self.head, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        x = self.attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.head * self.d_k)
        x = self.linears[-1](x)
        # returen b t dim
        return x

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = Linear(d_model, d_ff)
        self.w_2 = Linear(d_ff, d_model)
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.w_1(x), inplace=True)
        if self.dropout:
            h = self.dropout(h)
        return self.w_2(h)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    def forward(self, x, sublayer):
        if self.dropout:
            return self.norm(x + self.dropout(sublayer(x)))
        else:
            return self.norm(x + sublayer(x))

