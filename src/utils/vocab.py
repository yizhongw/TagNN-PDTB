#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: yizhong
# created_at: 17-5-2 下午3:47
import torch


class Vocab(object):
    def __init__(self, filename=None, mannual_add=None, lower=False):
        self.id2word = {}
        self.word2id = {}
        self.lower = lower
        self.embeddings = None
        self.embed_dim = None
        # set the mannual_add words to be zero initially
        self.zero_embedding_words = mannual_add

        if mannual_add is not None:
            for word in mannual_add:
                self.add(word)
        if filename is not None:
            self.load_file(filename)

    def size(self):
        return len(self.id2word)

    def load_file(self, filename):
        for line in open(filename, 'r'):
            token = line.rstrip('\n')
            self.add(token)

    def get_id(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.word2id[key]
        except KeyError:
            return default

    def get_word(self, idx, default=None):
        try:
            return self.id2word[idx]
        except KeyError:
            return default

    def add(self, label):
        label = label.lower() if self.lower else label
        if label in self.word2id:
            idx = self.word2id[label]
        else:
            idx = len(self.id2word)
            self.id2word[idx] = label
            self.word2id[label] = idx
        return idx

    def load_pretrained_emb(self, embedding_path):
        glove_embeddings = {}
        with open(embedding_path, 'r') as fin:
            for line in fin:
                contents = line.strip().split(' ')
                word = contents[0]
                glove_embeddings[word] = torch.Tensor(list(map(float, contents[1:])))
                if self.embed_dim is None:
                    self.embed_dim = len(contents) - 1
        filtered_words = set(self.word2id.keys()) & set(glove_embeddings.keys())
        # rebuild the word x id map
        self.word2id = {}
        self.id2word = {}
        for word in self.zero_embedding_words:
            self.add(word)
        for word in filtered_words:
            self.add(word)
        # load embeddings
        self.embeddings = torch.Tensor(self.size(), self.embed_dim)
        for word in self.zero_embedding_words:
            self.embeddings[self.get_id(word)].zero_()
        for word in self.word2id.keys():
            if word in glove_embeddings:
                self.embeddings[self.get_id(word)] = glove_embeddings[word]

    def convert_words2ids(self, words, unk_word, bos_word=None, eos_word=None):
        """Convert words to ids, use unk_word if the word is not in vocab."""
        vec = []
        if bos_word is not None:
            vec += [self.get_id(bos_word)]
        unk = self.get_id(unk_word)
        vec += [self.get_id(label, default=unk) for label in words]
        if eos_word is not None:
            vec += [self.get_id(eos_word)]
        return vec

    def convert_ids2words(self, ids, stop_id=None):
        words = []
        for i in ids:
            words += [self.get_word(i)]
            if stop_id is not None and i == stop_id:
                break
        return words
