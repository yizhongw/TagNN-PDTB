#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: yizhong
# created_at: 17-5-4 下午7:57
import torch
import torch.nn as nn


class BasicLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, bidirectional, use_cuda):
        super(BasicLSTM, self).__init__()
        self.use_cuda = use_cuda
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=1, bidirectional=bidirectional)

    def forward(self, inputs):
        embeds = torch.unsqueeze(self.emb(inputs), 1)
        outputs, (hn, cn) = self.rnn(embeds)
        if self.bidirectional:
            outputs = outputs.view(outputs.size(0), outputs.size(1), 2, -1)
            outputs = torch.sum(outputs, 2)
            outputs = outputs.squeeze(1)
            outputs = outputs.squeeze(1)
            hn = hn.sum(0)
            cn = cn.sum(0)
        return outputs, (cn.squeeze(0), hn.squeeze(0))


class BasicGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, bidirectional, use_cuda):
        super(BasicGRU, self).__init__()
        self.use_cuda = use_cuda
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(input_size=embed_dim, hidden_size=hidden_size, num_layers=1, bidirectional=bidirectional)

    def forward(self, inputs):
        embeds = torch.unsqueeze(self.emb(inputs), 1)
        outputs, hn = self.rnn(embeds)
        if self.bidirectional:
            outputs = outputs.view(outputs.size(0), outputs.size(1), 2, -1)
            outputs = torch.sum(outputs, 2)
            outputs = outputs.squeeze(1)
            outputs = outputs.squeeze(1)
            hn = hn.sum(0)
        return outputs, hn.squeeze(0)
