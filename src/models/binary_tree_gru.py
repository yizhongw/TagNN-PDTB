#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: yizhong
# created_at: 17-5-23 下午6:14
import torch
import torch.nn as nn
from torch.autograd import Variable


class BinaryTreeGRU(nn.Module):
    def __init__(self, word_vocab_size, word_embed_dim, hidden_size, use_cuda):
        super(BinaryTreeGRU, self).__init__()
        self.use_cuda = use_cuda
        self.word_embed_dim = word_embed_dim
        self.hidden_size = hidden_size

        self.word_embed_func = nn.Embedding(word_vocab_size, word_embed_dim)
        self.rg = nn.Linear(self.word_embed_dim + 2 * self.hidden_size, self.hidden_size)
        self.zg = nn.Linear(self.word_embed_dim + 2 * self.hidden_size, self.hidden_size)
        self.u = nn.Linear(self.word_embed_dim + 2 * self.hidden_size, self.hidden_size)

    def forward(self, root_node, inputs):
        outputs = []
        final_state = self.recursive_forward(root_node, inputs, outputs)
        outputs = torch.cat(outputs, 0)
        return outputs, final_state

    def recursive_forward(self, node, inputs, outputs):
        # get states from children
        child_states = []
        if len(node.children) == 0:
            left_h = Variable(torch.zeros(1, self.hidden_size))
            right_h = Variable(torch.zeros(1, self.hidden_size))
            if self.use_cuda:
                left_h, right_h = left_h.cuda(), right_h.cuda()
            child_states.append(left_h)
            child_states.append(right_h)
        else:
            assert len(node.children) <= 2
            for idx in range(len(node.children)):
                child_state = self.recursive_forward(node.children[idx], inputs, outputs)
                child_states.append(child_state)
        # calculate the state of current node
        node_state = self.node_forward(node, child_states)
        outputs.append(node_state)
        return node_state

    def node_forward(self, node, child_states):
        if node.idx is not None:
            node_word = node.word
            if self.use_cuda:
                node_word = node_word.cuda()
            word_embed = self.word_embed_func(node_word)
        else:
            word_embed = Variable(torch.zeros(1, self.word_embed_dim))
            if self.use_cuda:
                word_embed = word_embed.cuda()
        if len(child_states) == 1:
            return child_states[0]
        else:
            left_h, right_h = child_states
            r = torch.sigmoid(self.rg(torch.cat([word_embed, left_h, right_h], 1)))
            z = torch.sigmoid(self.zg(torch.cat([word_embed, left_h, right_h], 1)))
            u = torch.tanh(self.u(torch.cat([word_embed, torch.mul(r, left_h), torch.mul(r, right_h)], 1)))
            h = torch.mul(z, left_h + right_h) + torch.mul(1 - z, u)
            return h


class LabeledBinaryTreeGRU(nn.Module):
    def __init__(self, word_vocab_size, tag_vocab_size, word_embed_dim, tag_embed_dim, hidden_size, use_cuda):
        super(LabeledBinaryTreeGRU, self).__init__()
        self.use_cuda = use_cuda
        self.word_embed_dim = word_embed_dim
        self.tag_embed_dim = tag_embed_dim
        self.hidden_size = hidden_size

        self.word_embed_func = nn.Embedding(word_vocab_size, word_embed_dim)
        self.tag_embed_func = nn.Embedding(tag_vocab_size, tag_embed_dim)
        self.rg = nn.Linear(self.word_embed_dim + self.tag_embed_dim + 2 * self.hidden_size, self.hidden_size)
        self.zg = nn.Linear(self.word_embed_dim + self.tag_embed_dim + 2 * self.hidden_size, self.hidden_size)
        self.u = nn.Linear(self.word_embed_dim + 2 * self.hidden_size, self.hidden_size)

    def forward(self, root_node, inputs):
        outputs = []
        final_state = self.recursive_forward(root_node, inputs, outputs)
        outputs = torch.cat(outputs, 0)
        return outputs, final_state

    def recursive_forward(self, node, inputs, outputs):
        # get states from children
        child_states = []
        if len(node.children) == 0:
            left_h = Variable(torch.zeros(1, self.hidden_size))
            right_h = Variable(torch.zeros(1, self.hidden_size))
            if self.use_cuda:
                left_h, right_h = left_h.cuda(), right_h.cuda()
            child_states.append(left_h)
            child_states.append(right_h)
        else:
            assert len(node.children) <= 2
            for idx in range(len(node.children)):
                child_state = self.recursive_forward(node.children[idx], inputs, outputs)
                child_states.append(child_state)
        # calculate the state of current node
        node_state = self.node_forward(node, child_states)
        outputs.append(node_state)
        return node_state

    def node_forward(self, node, child_states):
        if node.idx is not None:
            node_word = node.word
            if self.use_cuda:
                node_word = node_word.cuda()
            word_embed = self.word_embed_func(node_word)
        else:
            word_embed = Variable(torch.zeros(1, self.word_embed_dim))
            if self.use_cuda:
                word_embed = word_embed.cuda()
        node_tag = node.tag
        if self.use_cuda:
            node_tag = node_tag.cuda()
        tag_embed = self.tag_embed_func(node_tag)
        if len(child_states) == 1:
            return child_states[0]
        else:
            left_h, right_h = child_states
            input = torch.cat([word_embed, tag_embed], 1)
            r = torch.sigmoid(self.rg(torch.cat([input, left_h, right_h], 1)))
            z = torch.sigmoid(self.zg(torch.cat([input, left_h, right_h], 1)))
            u = torch.tanh(self.u(torch.cat([word_embed, torch.mul(r, left_h), torch.mul(r, right_h)], 1)))
            h = torch.mul(z, left_h + right_h) + torch.mul(1 - z, u)
            return h
