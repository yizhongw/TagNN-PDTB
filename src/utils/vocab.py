#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: yizhong
# created_at: 17-5-2 下午3:47
import torch
import torch.nn.init as init


class Vocab(object):
    def __init__(self, filename=None, mannual_add=None, lower=False):
        self.id2token = {}
        self.token2id = {}
        self.lower = lower
        self.embeddings = None
        self.embed_dim = None
        # set the mannual_add tokens to be zero initially
        self.zero_embedding_tokens = mannual_add

        if mannual_add is not None:
            for token in mannual_add:
                self.add(token)
        if filename is not None:
            self.load_file(filename)

    def size(self):
        return len(self.id2token)

    def load_file(self, filename):
        for line in open(filename, 'r'):
            token = line.rstrip('\n')
            self.add(token)

    def get_id(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.token2id[key]
        except KeyError:
            return default

    def get_token(self, idx, default=None):
        try:
            return self.id2token[idx]
        except KeyError:
            return default

    def add(self, label):
        label = label.lower() if self.lower else label
        if label in self.token2id:
            idx = self.token2id[label]
        else:
            idx = len(self.id2token)
            self.id2token[idx] = label
            self.token2id[label] = idx
        return idx

    def init_embed(self, embed_dim):
        self.embed_dim = embed_dim
        self.embeddings = torch.Tensor(self.size(), embed_dim)
        init.xavier_normal(self.embeddings)

    def load_pretrained_emb(self, embedding_path):
        glove_embeddings = {}
        with open(embedding_path, 'r') as fin:
            for line in fin:
                contents = line.strip().split(' ')
                token = contents[0]
                glove_embeddings[token] = torch.Tensor(list(map(float, contents[1:])))
                if self.embed_dim is None:
                    self.embed_dim = len(contents) - 1
        filtered_tokens = set(self.token2id.keys()) & set(glove_embeddings.keys())
        # rebuild the token x id map
        self.token2id = {}
        self.id2token = {}
        for token in self.zero_embedding_tokens:
            self.add(token)
        for token in filtered_tokens:
            self.add(token)
        # load embeddings
        self.embeddings = torch.Tensor(self.size(), self.embed_dim)
        for token in self.zero_embedding_tokens:
            self.embeddings[self.get_id(token)].zero_()
        for token in self.token2id.keys():
            if token in glove_embeddings:
                self.embeddings[self.get_id(token)] = glove_embeddings[token]

    def convert2ids(self, tokens, unk_token, bos_token=None, eos_token=None):
        """Convert tokens to ids, use unk_token if the token is not in vocab."""
        vec = []
        if bos_token is not None:
            vec += [self.get_id(bos_token)]
        unk = self.get_id(unk_token)
        vec += [self.get_id(label, default=unk) for label in tokens]
        if eos_token is not None:
            vec += [self.get_id(eos_token)]
        return vec

    def recover_from_ids(self, ids, stop_id=None):
        tokens = []
        for i in ids:
            tokens += [self.get_token(i)]
            if stop_id is not None and i == stop_id:
                break
        return tokens
