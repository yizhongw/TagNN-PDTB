#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: yizhong
# created_at: 17-5-8 上午11:22
import torch
import torch.nn as nn


class AttentionNet(nn.Module):
    def __init__(self, att_vectors_dim, ref_vector_dim):
        super(AttentionNet, self).__init__()
        self.att_vectors_dim = att_vectors_dim
        self.ref_vector_dim = ref_vector_dim
        self.wh = nn.Linear(att_vectors_dim, att_vectors_dim, bias=False)
        self.wv = nn.Linear(ref_vector_dim, att_vectors_dim, bias=False)
        self.ws = nn.Linear(att_vectors_dim, 1, bias=False)
        self.softmax = nn.Softmax()

    def forward(self, att_vectors, ref_vector):
        h = torch.tanh(self.wh(att_vectors) + self.wv(ref_vector.expand(att_vectors.size(0), self.ref_vector_dim)))
        h = self.ws(h)
        h = h.view(1, -1)
        att_weights = self.softmax(h).squeeze(0)
        return attention_mul(att_vectors, att_weights)


def attention_mul(att_vectors, att_weights):
    attn_vectors = None
    for i in range(att_vectors.size(0)):
        h_i = att_vectors[i].unsqueeze(0)
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        if attn_vectors is None:
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors, h_i), 0)
    return torch.sum(attn_vectors, 0)
