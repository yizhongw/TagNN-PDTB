#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: yizhong
# created_at: 17-5-31 上午12:15
import os
import re
import pickle
import torch
from utils.tree import ConstTree, DepTree, ConstNode, DepNode
from utils.const import NUM_WORD, UNK_WORD
from torch.autograd import Variable


class PDTBInstance:
    def __init__(self):
        self.left_words = []
        self.right_words = []
        self.left_const_tree = None
        self.right_const_tree = None
        self.left_dep_tree = None
        self.right_dep_tree = None
        self.label = None

    def init_from_pipe_instance(self, pipe_instance):
        self.left_words, self.left_dep_tree, self.left_const_tree = extract_json_ob(pipe_instance.arg1_parse_result)
        self.right_words, self.right_dep_tree, self.right_const_tree = extract_json_ob(pipe_instance.arg2_parse_result)


class PDTBDataSet:
    def __init__(self, train_sections, dev_sections, test_sections, level=2):
        self.level = level
        self.train_set = self.load_instances(train_sections, multiple_labels=False)
        self.dev_set = self.load_instances(dev_sections)
        self.test_set = self.load_instances(test_sections)
        self.label_map = self._build_label_map()

    def load_instances(self, section_dirs, multiple_labels=True):
        instances = []
        for section_dir in section_dirs:
            for file in os.listdir(section_dir):
                if not file.endswith('pickle'):
                    continue
                with open(os.path.join(section_dir, file), 'rb') as fin:
                    pipe_inst = pickle.load(fin)
                    if pipe_inst.type != 'Implicit':
                        continue
                    if multiple_labels:
                        labels = set()
                        for sem in pipe_inst.sems:
                            label = get_subtype(sem, self.level)
                            if label is not None:
                                labels.add(label)
                        pdtb_inst = PDTBInstance()
                        pdtb_inst.init_from_pipe_instance(pipe_inst)
                        pdtb_inst.label = labels
                        instances.append(pdtb_inst)
                    else:
                        for sem in pipe_inst.sems:
                            label = get_subtype(sem, self.level)
                            if label is not None:
                                pdtb_inst = PDTBInstance()
                                pdtb_inst.init_from_pipe_instance(pipe_inst)
                                pdtb_inst.label = label
                                instances.append(pdtb_inst)
        return instances

    def _build_label_map(self):
        labels = set()
        for inst in self.train_set + self.dev_set + self.test_set:
            if isinstance(inst.label, str):
                labels.add(inst.label)
            else:
                for label in inst.label:
                    labels.add(label)
        label_map = {label: idx for idx, label in enumerate(sorted(labels))}
        print('{} labels in dataset'.format(len(label_map)))
        return label_map

    def get_all_words(self):
        words = []
        for inst in self.train_set + self.dev_set + self.test_set:
            words.extend(inst.left_words)
            words.extend(inst.right_words)
        return words

    def get_all_tags(self):
        tags = []
        for inst in self.train_set + self.dev_set + self.test_set:
            tags.extend([node.tag for node in inst.left_const_tree.bfs_tranverse()])
            tags.extend([node.tag for node in inst.right_const_tree.bfs_tranverse()])
        return tags

    def format_instances_to_torch_var(self, word_vocab=None, tag_vocab=None):
        for dataset in [self.train_set, self.dev_set, self.test_set]:
            for inst in dataset:
                inst.left_words = Variable(torch.LongTensor(word_vocab.convert2ids(inst.left_words, UNK_WORD)))
                inst.right_words = Variable(torch.LongTensor(word_vocab.convert2ids(inst.right_words, UNK_WORD)))
                if isinstance(inst.label, str):
                    inst.label = Variable(torch.LongTensor([self.label_map[inst.label]]))
                else:
                    inst.label = [self.label_map[label] for label in inst.label]
                if word_vocab is not None or tag_vocab is not None:
                    left_nodes = inst.left_const_tree.bfs_tranverse()
                    for node in left_nodes:
                        if word_vocab is not None and node.word is not None:
                            # don't use node.word, since it's the original, unprocessed text
                            node.word = inst.left_words[node.idx]
                        if tag_vocab is not None:
                            node.tag = Variable(torch.LongTensor([tag_vocab.get_id(node.tag)]))
                    right_nodes = inst.right_const_tree.bfs_tranverse()
                    for node in right_nodes:
                        if word_vocab is not None and node.word is not None:
                            # don't use node.word, since it's the original, unprocessed text
                            node.word = inst.right_words[node.idx]
                        if tag_vocab is not None:
                            node.tag = Variable(torch.LongTensor([tag_vocab.get_id(node.tag)]))


class PipeInstance:
    def __init__(self, columns):
        self.type = columns[0]
        self.section = columns[1]
        self.file = columns[2]
        self.arg1 = columns[24]
        self.arg2 = columns[34]
        self.connective = columns[5] # only for Explicit and AltLex
        self.connective_start = int(columns[3].split('..')[0]) if self.connective != '' else -1
        self.conn1 = columns[9] # only for Implicit
        self.conn2 = columns[10] # only for Implicit
        self.conn1_sem1 = columns[11] # only for Explicit, Implicit and AltLex
        self.conn1_sem2 = columns[12] # only for Explicit, Implicit and AltLex
        self.conn2_sem1 = columns[13] # only for Implicit
        self.conn2_sem2 = columns[14] # only for Implicit
        self.sems = list({self.conn1_sem1, self.conn1_sem2, self.conn2_sem1, self.conn1_sem2})
        self.sems.remove('')
        self.sem = self.conn1_sem1 # the dominant sem
        self.arg1_start = int(columns[22].split('..')[0])
        self.arg2_start = int(columns[32].split('..')[0])
        if self.arg1_start < self.arg2_start:
            self.direction = '<'
        else:
            self.direction = '>'
        # placeholder for the parse of sentence
        self.arg1_parse_result = None
        self.arg2_parse_result = None
        self.arg1_words = None
        self.arg2_words = None
        self.arg1_dep_tree = None
        self.arg2_dep_tree = None
        self.arg1_const_tree = None
        self.arg2_const_tree = None


def get_subtype(sem, level):
    sem_items = sem.split('.')
    if len(sem_items) < level:
        subtype = None
    else:
        subtype = sem_items[level-1]
    filter_sems_level_2 = {'Pragmatic contrast', 'Pragmatic concession',
                           'Condition', 'Pragmatic condition', 'Exception'}
    if level == 2 and subtype in filter_sems_level_2:
        subtype = None
    return subtype


def load_pipe_file(fpath, types=None):
    with open(fpath, 'r', encoding='ISO-8859-1') as fin:
        lines = fin.readlines()
    instances = []
    for line in lines:
        columns = line.split('|')
        instance = PipeInstance(columns)
        if types is not None and instance.type not in types:
            continue
        instances.append(instance)
    return instances


def extract_json_ob(json_ob, lowercase=True, use_lemma=True, replace_num=True):
    processed_words = []
    dep_trees = []
    const_trees = []
    # there may be multiple sentences
    for sentence_info in json_ob['sentences']:
        # extract the words of the sentence
        sent_words = []
        for token in sentence_info['tokens']:
            word = token['lemma'] if use_lemma else token['word']
            word = word.lower() if lowercase else word
            if replace_num and any(c.isdigit() for c in word):
                word = re.sub('[.|,|/| ]', '', word.lstrip('-'))
                if word.isdigit():
                    word = NUM_WORD
            sent_words.append(word)
        # build the dependency tree
        nodes = [DepNode(len(processed_words) + word_idx) for word_idx in range(len(sent_words))]
        root_node = None
        for dep in sentence_info['basicDependencies']:
            if dep['governor'] == 0:
                root_node = nodes[dep['dependent'] - 1]
            else:
                nodes[dep['governor'] - 1].add_child(nodes[dep['dependent'] - 1])
        dep_tree = DepTree()
        dep_tree.assign_root(root_node)
        dep_trees.append(dep_tree)
        # build the constituency tree
        const_tree = ConstTree()
        const_tree.load_from_string(sentence_info['parse'])
        const_tree.compress()
        const_tree.binarize()
        const_trees.append(const_tree)
        # add the sent_words to processed_words
        processed_words.extend(sent_words)
    if len(dep_trees) > 1:
        for dep_tree in dep_trees[1:]:
            dep_trees[0].merge(dep_tree)
    if len(const_trees) > 1:
        for const_tree in const_trees[1:]:
            const_trees[0].merge(const_tree)
        const_trees[0].binarize()
    return processed_words, dep_trees[0], const_trees[0]