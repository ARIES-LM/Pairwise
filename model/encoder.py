import torch
import numpy as np
import torch.nn as nn
from model.util import clone, MultiHeadedAttention, PositionwiseFeedForward, SublayerConnection


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clone(layer, N)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward"

    def __init__(self, d_model, n_heads, d_hidden, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(n_heads, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_hidden, dropout)

        self.sublayer = clone(SublayerConnection(d_model, dropout), 2)
        self.size = d_model

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

