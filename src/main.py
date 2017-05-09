#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: yizhong
# created_at: 17-5-2 下午3:02
import os
import torch
import torch.nn as nn
import torch.optim as optim
from config import parse_args, PathConfig, ModelConfig
from models.classifier import RelationClassifier
from models.trainer import Trainer
from utils.const import *
from utils.pdtb import PDTBDataSet
from utils.vocab import Vocab


def prepare_data():
    # load the dataset
    train_sections = [os.path.join(paths.json_data_dir, '{:02}'.format(section_num)) for section_num in
                      paths.train_sections]
    dev_sections = [os.path.join(paths.json_data_dir, '{:02}'.format(section_num)) for section_num in
                    paths.dev_sections]
    test_sections = [os.path.join(paths.json_data_dir, '{:02}'.format(section_num)) for section_num in
                     paths.test_sections]
    train_dataset, dev_dataset, test_dataset = PDTBDataSet(train_sections, tree_type=args.tree_type), \
                                               PDTBDataSet(dev_sections, tree_type=args.tree_type), \
                                               PDTBDataSet(test_sections, tree_type=args.tree_type)
    assert train_dataset.consistent_with(dev_dataset) and dev_dataset.consistent_with(test_dataset)
    print('Size of train set: {}, dev set: {}, test set: {}'.format(len(train_dataset), len(dev_dataset),
                                                                    len(test_dataset)))
    # save the dataset
    torch.save(train_dataset, os.path.join(paths.experiment_data_dir, 'train.data'))
    torch.save(dev_dataset, os.path.join(paths.experiment_data_dir, 'dev.data'))
    torch.save(test_dataset, os.path.join(paths.experiment_data_dir, 'test.data'))
    # build the vocab
    vocab = Vocab(mannual_add=[PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD, NUM_WORD])
    all_words = train_dataset.get_all_words() + dev_dataset.get_all_words() + test_dataset.get_all_words()
    for word in all_words:
        vocab.add(word)
    # load and initialize the embeddings
    vocab.load_pretrained_emb(paths.embedding_path)
    print('Size of PDTB vocabulary: {}'.format(vocab.size()))
    # save the vocab
    torch.save(vocab, paths.vocab_path)


def train_model():
    train_dataset = torch.load(os.path.join(paths.experiment_data_dir, 'train.data'))
    dev_dataset = torch.load(os.path.join(paths.experiment_data_dir, 'dev.data'))
    test_dataset = torch.load(os.path.join(paths.experiment_data_dir, 'test.data'))
    vocab = torch.load(paths.vocab_path)
    model = RelationClassifier(
        encoder_type=args.encoder,
        num_classes=len(train_dataset.label_map),
        vocab_size=vocab.size(),
        embed_dim=vocab.embed_dim,
        model_config=ModelConfig,
        use_cuda=args.cuda
    )
    # init with pre-trained embeddings
    model.argument_encoder.emb.weight.data.copy_(vocab.embeddings)
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        model.cuda(), criterion.cuda()
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'rprop':
        optimizer = optim.Rprop(model.parameters(), lr=args.lr)
    trainer = Trainer(model, criterion, optimizer, args.cuda)
    best_dev_f1 = 0
    for epoch in range(args.epochs):
        print()
        trainer.train(train_dataset, vocab, batch_size=args.batch_size)
        # train_loss, train_acc, train_f1 = trainer.eval(train_dataset, vocab, 'trainset')
        # print('Train: loss {}, acc {}, f1 {}'.format(train_loss, train_acc, train_f1))
        dev_loss, dev_acc, dev_f1 = trainer.eval(dev_dataset, vocab, 'devset')
        print('Dev: loss {}, acc {}, f1 {}'.format(dev_loss, dev_acc, dev_f1))
        test_loss, test_acc, test_f1 = trainer.eval(test_dataset, vocab, 'testset')
        print('Test: loss {}, acc {}, f1 {}'.format(test_loss, test_acc, test_f1))
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            print('Model saved to {}'.format(paths.best_model_path))
            torch.save(model.state_dict(), paths.best_model_path)


def test_model():
    train_dataset = torch.load(os.path.join(paths.experiment_data_dir, 'train.data'))
    dev_dataset = torch.load(os.path.join(paths.experiment_data_dir, 'dev.data'))
    test_dataset = torch.load(os.path.join(paths.experiment_data_dir, 'test.data'))
    vocab = torch.load(paths.vocab_path)
    model = RelationClassifier(
        encoder_type=args.encoder,
        num_classes=len(train_dataset.label_map),
        vocab_size=vocab.size(),
        embed_dim=vocab.embed_dim,
        model_config=ModelConfig,
        use_cuda=args.cuda
    )
    model.load_state_dict(torch.load(paths.best_model_path))
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        model.cuda(), criterion.cuda()
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wd)
    trainer = Trainer(model, criterion, optimizer, args.cuda)
    # train_loss, train_acc, train_f1 = trainer.eval(train_dataset, vocab, 'trainset')
    # print('Train: loss {}, acc {}, f1 {}'.format(train_loss, train_acc, train_f1))
    dev_loss, dev_acc, dev_f1 = trainer.eval(dev_dataset, vocab, 'devset')
    print('Dev: loss {}, acc {}, f1 {}'.format(dev_loss, dev_acc, dev_f1))
    test_loss, test_acc, test_f1 = trainer.eval(test_dataset, vocab, 'testset')
    print('Test: loss {}, acc {}, f1 {}'.format(test_loss, test_acc, test_f1))


def main():
    global args, paths
    args = parse_args()
    paths = PathConfig
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if args.prepare:
        prepare_data()
    if args.train:
        train_model()
    if args.test:
        test_model()


if __name__ == '__main__':
    main()
