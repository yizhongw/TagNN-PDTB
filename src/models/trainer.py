#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: yizhong
# created_at: 17-5-2 下午10:54
from sys import stdout
import time
import torch
from utils.const import UNK_WORD
from torch.autograd import Variable
from sklearn.metrics import f1_score


class Trainer:
    def __init__(self, model, criterion, optimizer, use_cuda):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.use_cuda = use_cuda
        self.epoch = 0

    def train(self, dataset, vocab, batch_size, dynamic_sample=False):
        self.model.train()
        self.optimizer.zero_grad()
        loss, k = 0.0, 0
        if dynamic_sample:
            train_perf_by_label = {label: {'correct': 0, 'total': 10, 'acc': 0}
                                   for label in range(self.model.num_classes)}
        else:
            indices = torch.randperm(len(dataset))
        start_time = time.strftime('%H:%M:%S', time.gmtime())
        for idx in range(len(dataset)):
            if dynamic_sample:
                least_perf_label = min(train_perf_by_label.keys(), key=lambda x: train_perf_by_label[x]['acc'])
                ltree, lsent, rtree, rsent, label = dataset.get_item_by_label(least_perf_label)
            else:
                ltree, lsent, rtree, rsent, label = dataset[indices[idx]]
            linput = Variable(torch.LongTensor(vocab.convert_words2ids(lsent, UNK_WORD)))
            rinput = Variable(torch.LongTensor(vocab.convert_words2ids(rsent, UNK_WORD)))
            target = Variable(torch.LongTensor([label]))
            if self.use_cuda:
                linput, rinput = linput.cuda(), rinput.cuda()
                target = target.cuda()
            pred, score = self.model(ltree, linput, rtree, rinput)
            err = self.criterion(score, target)
            loss += err.data[0]
            if dynamic_sample:
                correct = pred.data.eq(target.data).cpu().sum()
                train_perf_by_label[label]['correct'] += correct
                train_perf_by_label[label]['total'] += 1
                train_perf_by_label[label]['acc'] = train_perf_by_label[label]['correct'] / train_perf_by_label[label]['total']
            err.backward()
            k += 1
            if k % batch_size == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            stdout.write('\rTraining epoch {} start {} idx {} pred {} target {}'.format(self.epoch + 1,
                                                                                        start_time,
                                                                                        idx, pred.data[0][0],
                                                                                        target.data[0]))
        stdout.write('\n')
        self.epoch += 1
        return loss / len(dataset)

    def eval(self, dataset, vocab, dataset_name):
        self.model.eval()
        loss = 0
        correct = 0
        predictions = []
        gold_labels = []
        for idx in range(len(dataset)):
            ltree, lsent, rtree, rsent, label = dataset[idx]
            linput = Variable(torch.LongTensor(vocab.convert_words2ids(lsent, UNK_WORD)), volatile=True)
            rinput = Variable(torch.LongTensor(vocab.convert_words2ids(rsent, UNK_WORD)), volatile=True)
            target = Variable(torch.LongTensor([label]), volatile=True)
            if self.use_cuda:
                linput, rinput = linput.cuda(), rinput.cuda()
                target = target.cuda()
            pred, score = self.model(ltree, linput, rtree, rinput)
            err = self.criterion(score, target)
            loss += err.data[0]
            correct += pred.data.eq(target.data).cpu().sum()
            predictions.append(pred.data[0][0])
            gold_labels.append(label)
            stdout.write('\rEvaluating epoch {} on {} idx {} pred {} target {}'.format(self.epoch, dataset_name,
                                                                                       idx, pred.data[0][0], target.data[0]))
        stdout.write('\n')
        return loss / len(dataset), correct / len(dataset), f1_score(gold_labels, predictions, average='macro')
