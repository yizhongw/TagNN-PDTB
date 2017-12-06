#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: yizhong
# created_at: 17-5-23 下午6:15
import torch
import torch.nn as nn


class BasicRecursiveNN(nn.Module):
    def __init__(self, word_vocab_size, word_embed_dim, hidden_size, use_cuda):
        super(BasicRecursiveNN, self).__init__()
        self.use_cuda = use_cuda
        self.word_embed_dim = word_embed_dim
        self.hidden_size = hidden_size
        assert word_embed_dim == hidden_size
        self.word_embed_func = nn.Embedding(word_vocab_size, word_embed_dim)
        self.fc = nn.Linear(2 * word_embed_dim, hidden_size)

    def forward(self, root_node, inputs):
        outputs = []
        final_state = self.recursive_forward(root_node.children[0], inputs, outputs)
        outputs = torch.cat(outputs, 0)
        return outputs, final_state

    def recursive_forward(self, node, inputs, outputs):
        # get states from children
        if len(node.children) == 0:
            node_word = node.word
            if self.use_cuda:
                node_word = node_word.cuda()
            word_embed = self.word_embed_func(node_word)
            outputs.append(word_embed)
            return word_embed
        else:
            assert len(node.children) <= 2
            child_states = []
            for idx in range(len(node.children)):
                child_state = self.recursive_forward(node.children[idx], inputs, outputs)
                child_states.append(child_state)
            if len(child_states) == 1:
                return child_states[0]
            else:
                node_state = torch.tanh(self.fc(torch.cat([child_states[0], child_states[1]], 1)))
                outputs.append(node_state)
                return node_state