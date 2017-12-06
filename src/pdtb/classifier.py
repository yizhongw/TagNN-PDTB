#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: yizhong
# created_at: 17-5-31 上午10:07

import torch
import torch.nn as nn
from models.child_sum_rnn import ChildSumTreeLSTM
from models.basic_recursive_nn import BasicRecursiveNN
from models.binary_tree_lstm import BinaryTreeLSTM, LabeledBinaryTreeLSTM
from models.binary_tree_gru import BinaryTreeGRU, LabeledBinaryTreeGRU
from models.basic_recurrent import BasicLSTM, BasicGRU


class RelationClassifier(nn.Module):
    def __init__(self, encoder_type, num_classes, word_vocab_size, tag_vocab_size, word_embed_dim, tag_embed_dim,
                 model_config, drop_rate, use_cuda=True):
        super(RelationClassifier, self).__init__()
        self.encoder_type = encoder_type
        self.num_classes = num_classes
        self.use_cuda = use_cuda
        self.dropout = nn.Dropout(drop_rate)
        self.argument_encoder = self.get_argument_encoder(encoder_type, word_vocab_size, tag_vocab_size,
                                                          word_embed_dim, tag_embed_dim,
                                                          model_config.lstm_hidden_size, use_cuda)
        self.output_linear = nn.Linear(model_config.lstm_hidden_size * 2, num_classes)

    def forward(self, left_words, right_words, inst):
        if self.encoder_type == 'child-sum-lstm':
            left_tree, right_tree = inst.left_dep_tree, inst.right_dep_tree
        else:
            left_tree, right_tree = inst.left_const_tree, inst.right_const_tree
        left_outputs, (left_state, left_hidden) = self.encode_argument(left_tree, left_words)
        right_outputs, (right_state, right_hidden) = self.encode_argument(right_tree, right_words)
        score = self.output_layer(left_hidden, right_hidden)
        _, pred = torch.max(score, 1)
        return pred, score

    def output_layer(self, left_h, right_h):
        score = self.output_linear(torch.cat([left_h, right_h], 1))
        return score

    def encode_argument(self, tree, words):
        if self.encoder_type in ['child-sum-lstm', 'binary-tree-lstm', 'labeled-binary-tree-lstm']:
            outputs, (state, hidden) = self.argument_encoder(tree.root, words)
        elif self.encoder_type in ['binary-tree-gru', 'labeled-binary-tree-gru']:
            outputs, hidden = self.argument_encoder(tree.root, words)
            state = None
        elif self.encoder_type == 'lstm' or self.encoder_type == 'bi-lstm':
            outputs, (state, hidden) = self.argument_encoder(words)
        elif self.encoder_type == 'gru' or self.encoder_type == 'bi-gru':
            outputs, hidden = self.argument_encoder(words)
            state = None
        elif self.encoder_type == 'recursive-nn':
            outputs, hidden = self.argument_encoder(tree.root, words)
            state = None
        else:
            raise NotImplementedError('Unsupported encoder type {}'.format(self.encoder_type))
        return outputs, (state, hidden)

    def get_argument_encoder(self, encoder_type, word_vocab_size, tag_vocab_size, word_embed_dim, tag_embed_dim, lstm_hidden_size, use_cuda):
        if encoder_type == 'child-sum-lstm':
            argument_encoder = ChildSumTreeLSTM(word_vocab_size, word_embed_dim, lstm_hidden_size, use_cuda)
        elif encoder_type == 'binary-tree-lstm':
            argument_encoder = BinaryTreeLSTM(word_vocab_size, word_embed_dim, lstm_hidden_size, use_cuda)
        elif encoder_type == 'labeled-binary-tree-lstm':
            argument_encoder = LabeledBinaryTreeLSTM(word_vocab_size, tag_vocab_size,
                                                     word_embed_dim, tag_embed_dim, lstm_hidden_size, use_cuda)
        elif encoder_type == 'binary-tree-gru':
            argument_encoder = BinaryTreeGRU(word_vocab_size, word_embed_dim, lstm_hidden_size, use_cuda)
        elif encoder_type == 'labeled-binary-tree-gru':
            argument_encoder = LabeledBinaryTreeGRU(word_vocab_size, tag_vocab_size,
                                                     word_embed_dim, tag_embed_dim, lstm_hidden_size, use_cuda)
        elif encoder_type == 'lstm' or encoder_type == 'bi-lstm':
            argument_encoder = BasicLSTM(word_vocab_size, word_embed_dim, lstm_hidden_size,
                                         bidirectional=encoder_type.startswith('bi'), use_cuda=use_cuda)
        elif encoder_type == 'gru' or encoder_type == 'bi-gru':
            argument_encoder = BasicGRU(word_vocab_size, word_embed_dim, lstm_hidden_size,
                                        bidirectional=encoder_type.startswith('bi'), use_cuda=use_cuda)
        elif encoder_type == 'recursive-nn':
            argument_encoder = BasicRecursiveNN(word_vocab_size, word_embed_dim, lstm_hidden_size, use_cuda=use_cuda)
        else:
            raise NotImplementedError('Unsupported encoder type {}'.format(encoder_type))
        return argument_encoder
