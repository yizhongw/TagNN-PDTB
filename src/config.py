#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: yizhong
# created_at: 17-5-2 下午3:03
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='PDTB implicit relation classification')
    parser.add_argument('--prepare', action='store_true', help='extract and convert the data')
    parser.add_argument('--train', action='store_true', help='train the model')
    parser.add_argument('--test', action='store_true', help='test the model')
    prepare_parser = parser.add_argument_group('prepare settings')
    prepare_parser.add_argument('--tree_type', choices=['dependency', 'constituency'], default='dependency',
                                help='which kind of tree type to use in dataset')
    prepare_parser.add_argument('--level', type=int, choices=[1, 2, 3], default=1,
                                help='the relation level')
    train_parser = parser.add_argument_group('train settings')
    train_parser.add_argument('--encoder',
                              choices=['lstm', 'bi-lstm', 'gru', 'bi-gru', 'child-sum-lstm', 'binary-tree-lstm'],
                              default='child-sum-lstm', help='argument encoder type')
    train_parser.add_argument('--attention', action='store_true', help='use attention')
    train_parser.add_argument('--batch_size', default=25, type=int,
                              help='batch size for optimizer updates')
    train_parser.add_argument('--epochs', default=15, type=int,
                              help='number of total epochs to run')
    train_parser.add_argument('--lr', default=0.01, type=float,
                              metavar='LR', help='initial learning rate')
    train_parser.add_argument('--wd', default=1e-4, type=float,
                              help='weight decay (default: 1e-4)')
    train_parser.add_argument('--optim', default='adagrad',
                              help='optimizer (default: adagrad)')
    train_parser.add_argument('--seed', default=123, type=int,
                              help='random seed (default: 123)')
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true', help='use GPU cuda')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='not use GPU cuda')
    parser.set_defaults(cuda=True)
    return parser.parse_args()


class PathConfig:
    pipe_data_dir = '/home/yizhong/Workspace/Discourse/data/pdtb_v2/pipe'
    json_data_dir = '/home/yizhong/Workspace/Discourse/data/pdtb_v2/converted'
    experiment_data_dir = '../data'
    train_sections = set(list(range(2, 21)))
    dev_sections = {0, 1}
    test_sections = {21, 22}
    vocab_path = experiment_data_dir + '/vocab.data'
    embedding_path = '/home/yizhong/Workspace/Discourse/data/embeddings/glove.6B.50d.txt'
    best_model_path = experiment_data_dir + '/model.info'


class ModelConfig:
    lstm_hidden_size = 250
    merge_output_size = 300
