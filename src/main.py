#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: yizhong
# created_at: 17-5-23 下午3:54
import argparse
from pdtb.run import *


def parse_args():
    parser = argparse.ArgumentParser(description='Tag-Enhanced TreeNN Models.')
    parser.add_argument('--preprocess', action='store_true',
                        help='tokenize and parse the pdtb pipe data')
    parser.add_argument('--prepare', action='store_true',
                        help='prepare the vocab and format the data')
    parser.add_argument('--train', action='store_true',
                        help='train and evaluate the model')
    task_parser = parser.add_argument_group('tasks')
    task_parser.add_argument('--task', choices=['coarse-pdtb', 'fine-pdtb'],
                             default='coarse-pdtb', help='which task')
    train_parser = parser.add_argument_group('train settings')
    train_parser.add_argument('--encoder',
                              choices=['lstm', 'bi-lstm', 'gru', 'bi-gru', 'recursive-nn',
                                       'child-sum-lstm', 'binary-tree-lstm', 'binary-tree-gru',
                                       'labeled-binary-tree-lstm', 'labeled-binary-tree-gru'],
                              default='labeled-binary-tree-lstm', help='argument encoder type')
    train_parser.add_argument('--batch_size', default=10, type=int,
                              help='batch size for optimizer updates')
    train_parser.add_argument('--epochs', default=15, type=int,
                              help='number of total epochs to run')
    train_parser.add_argument('--lr', default=0.01, type=float,
                              metavar='LR', help='initial learning rate')
    train_parser.add_argument('--wd', default=1e-4, type=float,
                              help='weight decay (default: 1e-4)')
    train_parser.add_argument('--drop', default=0, type=float,
                              help='dropout rate')
    train_parser.add_argument('--optim', default='adagrad',
                              help='optimizer (default: adagrad)')
    train_parser.add_argument('--seed', default=123, type=int,
                              help='random seed (default: 123)')
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true', help='use GPU cuda')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='not use GPU cuda')
    parser.set_defaults(cuda=True)
    cuda_parser.add_argument('--gpu', default='0', help='the gpu device to use')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print('Run with args: {}'.format(args))

    torch.manual_seed(args.seed)
    if args.cuda:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        torch.cuda.manual_seed(args.seed)

    if args.task.endswith('pdtb'):
        if args.preprocess:
            pdtb_preprocess(args)
        if args.prepare:
            pdtb_prepare(args)
        if args.train:
            pdtb_train(args)

