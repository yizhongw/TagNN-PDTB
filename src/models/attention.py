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
        ref_vectors = ref_vector.expand(att_vectors.size(0), self.ref_vector_dim)
        h = torch.tanh(self.wh(att_vectors) + self.wv(ref_vectors))
        h = self.ws(h)
        h = h.view(1, -1)
        att_weights = self.softmax(h)
        return torch.mm(att_weights, att_vectors)


# class AttentionNet(nn.Module):
#     def __init__(self, att_vectors_dim, ref_vector_dim):
#         super(AttentionNet, self).__init__()
#         self.att_vectors_dim = att_vectors_dim
#         self.ref_vector_dim = ref_vector_dim
#         self.wr = nn.Linear(ref_vector_dim, att_vectors_dim)
#         self.softmax = nn.Softmax()
#
#     def forward(self, att_vectors, ref_vector):
#         h = self.wr(ref_vector)
#         h = torch.mm(att_vectors, torch.transpose(h, 0, 1))
#         h = h.view(1, -1)
#         att_weights = self.softmax(h)
#         return torch.mm(att_weights, att_vectors)