#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: yizhong
# created_at: 17-5-2 下午8:33
import torch
import torch.nn as nn
from models.child_sum_rnn import ChildSumTreeLSTM
from models.binary_tree_rnn import BinaryTreeLSTM
from models.basic_recurrent import BasicLSTM, BasicGRU
from models.attention import AttentionNet


class RelationClassifier(nn.Module):
    def __init__(self, encoder_type, num_classes, vocab_size, embed_dim, model_config, use_cuda=True, attention=False):
        super(RelationClassifier, self).__init__()
        self.encoder_type = encoder_type
        self.num_classes = num_classes
        self.use_cuda = use_cuda
        self.attention = attention
        self.argument_encoder = self.get_argument_encoder(encoder_type, vocab_size, embed_dim,
                                                          model_config.lstm_hidden_size, use_cuda)
        if self.attention:
            self.attention_net = AttentionNet(att_vectors_dim=model_config.lstm_hidden_size,
                                              ref_vector_dim=model_config.lstm_hidden_size)
        self.merge_fc = nn.Sequential(nn.Linear(3 * model_config.lstm_hidden_size, model_config.merge_output_size),
                                      nn.ReLU())
        self.final_fc = nn.Linear(model_config.merge_output_size, num_classes)

    def forward(self, arg1_tree, arg1_words, arg2_tree, arg2_words):
        arg1_outputs, arg1_hidden = self.encode_argument(arg1_tree, arg1_words)
        arg2_outputs, arg2_hidden = self.encode_argument(arg2_tree, arg2_words)
        if self.attention:
            arg1_att = self.attention_net(arg1_outputs, arg2_hidden)
            arg2_att = self.attention_net(arg2_outputs, arg1_hidden)
            arg12_merge = self.merge_fc(torch.cat([arg1_att, arg2_att, arg1_att - arg2_att], 1))
        else:
            arg12_merge = self.merge_fc(torch.cat([arg1_hidden, arg2_hidden, arg1_hidden - arg2_hidden], 1))
        score = self.final_fc(arg12_merge)
        _, pred = torch.max(score, 1)
        return pred, score

    def encode_argument(self, tree, words):
        if self.encoder_type == 'child-sum-lstm' or self.encoder_type == 'binary-tree-lstm':
            outputs, (state, hidden) = self.argument_encoder(tree.root, words)
        elif self.encoder_type == 'lstm' or self.encoder_type == 'bi-lstm':
            outputs, (state, hidden) = self.argument_encoder(words)
        elif self.encoder_type == 'gru' or self.encoder_type == 'bi-gru':
            outputs, hidden = self.argument_encoder(words)
        else:
            raise NotImplementedError('Unsupported encoder type {}'.format(self.encoder_type))
        return outputs, hidden

    def get_argument_encoder(self, encoder_type, vocab_size, embed_dim, lstm_hidden_size, use_cuda):
        if encoder_type == 'child-sum-lstm':
            argument_encoder = ChildSumTreeLSTM(vocab_size, embed_dim, lstm_hidden_size, use_cuda)
        elif encoder_type == 'binary-tree-lstm':
            argument_encoder = BinaryTreeLSTM(vocab_size, embed_dim, lstm_hidden_size, use_cuda)
        elif encoder_type == 'lstm' or encoder_type == 'bi-lstm':
            argument_encoder = BasicLSTM(vocab_size, embed_dim, lstm_hidden_size,
                                         bidirectional=encoder_type.startswith('bi'), use_cuda=use_cuda)
        elif encoder_type == 'gru' or encoder_type == 'bi-gru':
            argument_encoder = BasicGRU(vocab_size, embed_dim, lstm_hidden_size,
                                        bidirectional=encoder_type.startswith('bi'), use_cuda=use_cuda)
        else:
            raise NotImplementedError('Unsupported encoder type {}'.format(encoder_type))
        return argument_encoder
