#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: yizhong
# created_at: 17-5-23 下午4:11
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from pycorenlp import StanfordCoreNLP
from pdtb.config import PathConfig, ModelConfig
from pdtb.dataset import load_pipe_file, PDTBDataSet
from utils.const import *
from utils.vocab import Vocab
from pdtb.classifier import RelationClassifier
from pdtb.trainer import Trainer


def pdtb_preprocess(args):
    sections = os.listdir(PathConfig.pipe_data_dir)
    if not os.path.exists(PathConfig.json_data_dir):
        os.mkdir(PathConfig.json_data_dir)
    core_nlp = StanfordCoreNLP('http://localhost:9000')
    annotate_func = lambda x: core_nlp.annotate(x, properties={
        'annotators': 'tokenize,ssplit,pos,lemma,parse,depparse',
        'outputFormat': 'json',
        # 'ssplit.isOneSentence': True
    })
    instance_cnt = 0
    for section in sections:
        raw_sec_dir = os.path.join(PathConfig.pipe_data_dir, section)
        if not os.path.isdir(raw_sec_dir):
            continue
        converted_sec_dir = os.path.join(PathConfig.json_data_dir, section)
        if not os.path.exists(converted_sec_dir):
            os.mkdir(converted_sec_dir)
        for file in os.listdir(raw_sec_dir):
            fpath = os.path.join(raw_sec_dir, file)
            pipe_instances = load_pipe_file(fpath, types=['Implicit'])
            basename_prefix = os.path.basename(fpath).split('.')[0]
            for idx, inst in enumerate(pipe_instances, 1):
                inst.arg1_parse_result = annotate_func(inst.arg1)
                inst.arg2_parse_result = annotate_func(inst.arg2)
                with open(os.path.join(converted_sec_dir, '{}.{}.pickle'.format(basename_prefix, idx)), 'wb') as fout:
                    pickle.dump(inst, fout)
                instance_cnt += 1
                if instance_cnt % 100 == 0:
                    print(instance_cnt)
    print('Totally, {} instances are converted.'.format(instance_cnt))


def pdtb_prepare(args):
    print('Loading dataset...')
    train_sections = [os.path.join(PathConfig.json_data_dir, '{:02}'.format(section_num)) for section_num in
                      PathConfig.train_sections]
    dev_sections = [os.path.join(PathConfig.json_data_dir, '{:02}'.format(section_num)) for section_num in
                    PathConfig.dev_sections]
    test_sections = [os.path.join(PathConfig.json_data_dir, '{:02}'.format(section_num)) for section_num in
                     PathConfig.test_sections]
    dataset = PDTBDataSet(train_sections, dev_sections, test_sections, level=2 if args.task.startswith('fine') else 1)
    print('Size of train: {}, dev: {}, test: {}'.format(len(dataset.train_set), len(dataset.dev_set),
                                                        len(dataset.test_set)))
    print('Creating word vocab...')
    if not os.path.exists(PathConfig.experiment_data_dir):
        os.mkdir(PathConfig.experiment_data_dir)
    word_vocab = Vocab(mannual_add=[PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD, NUM_WORD])
    for word in dataset.get_all_words():
        word_vocab.add(word)
    word_vocab.load_pretrained_emb(PathConfig.embedding_path)
    print('Size of word vocab: {}'.format(word_vocab.size()))
    torch.save(word_vocab, os.path.join(PathConfig.experiment_data_dir, 'word_vocab.obj'))
    tag_vocab = Vocab()
    for tag in dataset.get_all_tags():
        tag_vocab.add(tag)
    print('Size of tag vocab: {}'.format(tag_vocab.size()))
    tag_vocab.init_embed(ModelConfig.tag_embed_dim)
    torch.save(tag_vocab, os.path.join(PathConfig.experiment_data_dir, 'tag_vocab.obj'))
    print('Formatting the dataset to torch variables...')
    dataset.format_instances_to_torch_var(word_vocab, tag_vocab)
    torch.save(dataset, os.path.join(PathConfig.experiment_data_dir, 'dataset.obj'))


def pdtb_train(args):
    print('Loading the data...')
    dataset = torch.load(os.path.join(PathConfig.experiment_data_dir, 'dataset.obj'))
    word_vocab = torch.load(os.path.join(PathConfig.experiment_data_dir, 'word_vocab.obj'))
    tag_vocab = torch.load(os.path.join(PathConfig.experiment_data_dir, 'tag_vocab.obj'))
    print('Initialize the model')
    model = RelationClassifier(
        encoder_type=args.encoder,
        num_classes=len(dataset.label_map),
        word_vocab_size=word_vocab.size(),
        tag_vocab_size=tag_vocab.size(),
        word_embed_dim=word_vocab.embed_dim,
        tag_embed_dim=tag_vocab.embed_dim,
        model_config=ModelConfig,
        drop_rate=args.drop,
        use_cuda=args.cuda
    )
    print('Model to train:')
    print(model)
    # init with pre-trained embeddings
    model.argument_encoder.word_embed_func.weight.data.copy_(word_vocab.embeddings)
    # model.argument_encoder.emb.weight.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    params_to_train = filter(lambda p: p.requires_grad, model.parameters())
    if args.cuda:
        model.cuda(), criterion.cuda()
    optimizer = get_optimizer(args.optim, params_to_train, args.lr, args.wd)
    trainer = Trainer(model, criterion, optimizer, args.cuda)
    global best_test_acc
    best_test_acc = 0
    def eval_func():
        print()
        global best_test_acc
        dev_acc = trainer.eval(dataset.dev_set, 'devset')
        print('Dev: acc {}'.format(dev_acc))
        test_acc = trainer.eval(dataset.test_set, 'testset')
        print('Test: acc {}'.format(test_acc))
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print('Model saved to {}'.format(PathConfig.best_model_path))
            torch.save(model.state_dict(), PathConfig.best_model_path)

    for epoch in range(args.epochs):
        print()
        trainer.train(dataset.train_set, batch_size=args.batch_size, eval_every_n_batch=300, eval_func=eval_func)
        # train_loss, train_acc, train_f1 = trainer.eval(train_dataset, vocab, 'trainset')
        # print('Train: loss {}, acc {}, f1 {}'.format(train_loss, train_acc, train_f1))
        # eval_func()


def get_optimizer(type, params_to_train, lr, wd):
    if type == 'sgd':
        optimizer = optim.SGD(params_to_train, lr=lr, weight_decay=wd)
    elif type == 'adam':
        optimizer = optim.Adam(params_to_train, lr=lr, weight_decay=wd)
    elif type == 'adagrad':
        optimizer = optim.Adagrad(params_to_train, lr=lr, weight_decay=wd)
    elif type == 'rprop':
        optimizer = optim.Rprop(params_to_train, lr=lr)
    else:
        raise NotImplementedError('Unsupported optimizer type.')
    return optimizer
