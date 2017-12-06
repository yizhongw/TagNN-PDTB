#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: yizhong
# created_at: 17-5-31 上午10:07

import torch
from utils.other import progress_bar


class Trainer:
    def __init__(self, model, criterion=None, optimizer=None, use_cuda=True):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.use_cuda = use_cuda
        self.epoch = 0

    def train(self, train_set, batch_size, eval_every_n_batch=None, eval_func=None):
        self.model.train()
        self.optimizer.zero_grad()
        loss, k, batch_cnt = 0.0, 0, 0
        indices = torch.randperm(len(train_set))
        for idx in range(len(train_set)):
            inst = train_set[indices[idx]]
            left_input = inst.left_words
            right_input = inst.right_words
            target = inst.label
            if self.use_cuda:
                left_input, right_input, target = left_input.cuda(), right_input.cuda(), target.cuda()
            pred, score = self.model(left_input, right_input, inst)
            err = self.criterion(score, target)
            # divide err by batch_size so that err.backward() accumulate the average gradients
            err = err / batch_size
            loss += err.data[0]
            err.backward()
            k += 1
            if k % batch_size == 0 or idx == len(train_set) - 1:
                self.optimizer.step()
                self.optimizer.zero_grad()
                batch_cnt += 1
                if eval_every_n_batch is not None and eval_func is not None and (
                                batch_cnt % eval_every_n_batch == 0 or idx == len(train_set) - 1):
                    self.model.eval()
                    eval_func()
                    self.model.train()
            progress_bar(k, len(train_set),
                         msg='Train epoch {} pred {} target {}'.format(self.epoch + 1, pred.data[0][0], target.data[0]))
        self.epoch += 1
        return loss / len(train_set)

    def eval(self, test_set, dataset_name):
        self.model.eval()
        correct = 0
        predictions = []
        gold_labels = []
        for idx in range(len(test_set)):
            inst = test_set[idx]
            left_input = inst.left_words
            right_input = inst.right_words
            left_input.volatile = True
            right_input.volatile = True
            if self.use_cuda:
                left_input, right_input = left_input.cuda(), right_input.cuda()
            pred, score = self.model(left_input, right_input, inst)
            predictions.append(pred.data[0][0])
            gold_labels.append(inst.label)
            if pred.data[0][0] in gold_labels[-1]:
                correct += 1
            progress_bar(idx, len(test_set),
                         msg='Eval epoch {} on {} pred {} target {}'.format(self.epoch, dataset_name,
                                                                            pred.data[0][0], gold_labels[-1]))
        return correct / len(test_set)