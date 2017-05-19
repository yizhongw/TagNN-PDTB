#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: yizhong
# created_at: 17-5-2 下午10:54
import torch
from utils.const import UNK_WORD
from torch.autograd import Variable
from utils.other import progress_bar


class Trainer:
    def __init__(self, model, criterion=None, optimizer=None, use_cuda=True):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.use_cuda = use_cuda
        self.epoch = 0

    def train(self, dataset, vocab, batch_size):
        self.model.train()
        self.optimizer.zero_grad()
        loss, k = 0.0, 0
        indices = torch.randperm(len(dataset))
        for idx in range(len(dataset)):
            ltree, lsent, rtree, rsent, label = dataset[indices[idx]]
            linput = Variable(torch.LongTensor(vocab.convert_words2ids(lsent, UNK_WORD)))
            rinput = Variable(torch.LongTensor(vocab.convert_words2ids(rsent, UNK_WORD)))
            target = Variable(torch.LongTensor([label]))
            if self.use_cuda:
                linput, rinput = linput.cuda(), rinput.cuda()
                target = target.cuda()
            pred, score = self.model(ltree, linput, rtree, rinput)
            err = self.criterion(score, target)
            # divide err by batch_size so that err.backward() accumulate the average gradients
            err = err / batch_size
            loss += err.data[0]
            err.backward()
            k += 1
            if k % batch_size == 0 or idx == len(dataset) - 1:
                self.optimizer.step()
                self.optimizer.zero_grad()
            progress_bar(idx, len(dataset),
                         msg='Train epoch {} pred {} target {}'.format(self.epoch + 1, pred.data[0][0], target.data[0]))
        self.epoch += 1
        return loss / len(dataset)

    def eval(self, dataset, vocab, dataset_name, multiple_labels=True):
        self.model.eval()
        correct = 0
        predictions = []
        gold_labels = []
        for idx in range(len(dataset)):
            ltree, lsent, rtree, rsent, label = dataset[idx]
            linput = Variable(torch.LongTensor(vocab.convert_words2ids(lsent, UNK_WORD)), volatile=True)
            rinput = Variable(torch.LongTensor(vocab.convert_words2ids(rsent, UNK_WORD)), volatile=True)
            if self.use_cuda:
                linput, rinput = linput.cuda(), rinput.cuda()
            pred, score = self.model(ltree, linput, rtree, rinput)
            predictions.append(pred.data[0][0])
            gold_labels.append(label)
            if multiple_labels:
                if pred.data[0][0] in label:
                    correct += 1
            else:
                if pred.data[0][0] == label:
                    correct += 1
            progress_bar(idx, len(dataset),
                         msg='Eval epoch {} on {} pred {} target {}'.format(self.epoch, dataset_name,
                                                                            pred.data[0][0], ','.join(map(str, label))))
        return correct / len(dataset)
