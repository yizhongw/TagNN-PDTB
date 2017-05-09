#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: yizhong
# created_at: 17-4-21 下午9:12
import os
import pickle
import re
from utils.tree import DepNode, DepTree, ConstTree
from utils.const import NUM_WORD
from pycorenlp import StanfordCoreNLP
from utils.pdtb import PDTBInstance

instance_cnt = 0

def load_pipe_file(fpath, level, types=None):
    with open(fpath, 'r', encoding='ISO-8859-1') as fin:
        lines = fin.readlines()
    instances = []
    for line in lines:
        columns = line.split('|')
        instance = PDTBInstance(columns, level)
        if types is not None and instance.type not in types:
            continue
        instances.append(instance)
    return instances


def convert_one_file(fpath, save_dir, arg_annotate_func):
    pdtb_instances = load_pipe_file(fpath, level=3, types=['Implicit'])
    basename_prefix = os.path.basename(fpath).split('.')[0]
    global instance_cnt
    for idx, inst in enumerate(pdtb_instances, 1):
        inst.arg1_parse_result = arg_annotate_func(inst.arg1)
        inst.arg2_parse_result = arg_annotate_func(inst.arg2)
        inst.arg1_words, inst.arg1_dep_tree, inst.arg1_const_tree = extract_from_json_ob(inst.arg1_parse_result)
        inst.arg2_words, inst.arg2_dep_tree, inst.arg2_const_tree = extract_from_json_ob(inst.arg2_parse_result)
        # assert that the tree size is compatible
        assert len(inst.arg1_words) == inst.arg1_dep_tree.get_size() == inst.arg1_const_tree.leaf_num <= inst.arg1_const_tree.get_size()
        assert len(inst.arg2_words) == inst.arg2_dep_tree.get_size() == inst.arg2_const_tree.leaf_num <= inst.arg2_const_tree.get_size()
        with open(os.path.join(save_dir, '{}.{}.pickle'.format(basename_prefix, idx)), 'wb') as fout:
            pickle.dump(inst, fout)
        instance_cnt += 1
        if instance_cnt % 100 == 0:
            print(instance_cnt)


def extract_from_json_ob(json_ob, lowercase=True, use_lemma=True, replace_num=True):
    sentence_info = json_ob['sentences'][0]
    # extract the words of the sentence
    words = []
    for token in sentence_info['tokens']:
        word = token['lemma'] if use_lemma else token['word']
        word = word.lower() if lowercase else word
        if replace_num and any(c.isdigit() for c in word):
            word = re.sub('[.|,|/| ]', '', word.lstrip('-'))
            if word.isdigit():
                word = NUM_WORD
        words.append(word)
    # build the dependency tree
    nodes = [DepNode(word_idx) for word_idx in range(len(words))]
    root_node = None
    for dep in sentence_info['basicDependencies']:
        if dep['governor'] == 0:
            root_node = nodes[dep['dependent']-1]
        else:
            nodes[dep['governor']-1].add_child(nodes[dep['dependent']-1])
    dep_tree = DepTree()
    dep_tree.assign_root(root_node)
    # build the constituency tree
    const_tree = ConstTree()
    const_tree.load_from_string(sentence_info['parse'])
    const_tree.compress()
    const_tree.binarize()
    return words, dep_tree, const_tree


def extract_from_conll_str(conll_str, lowercase=True, use_lemma=True, replace_num=True):
    words = []
    nodes = []
    tokens = [line.strip().split('\t') for line in conll_str.strip().split('\n') if line.strip()]
    # init the word and nodes
    for token_idx, token_info in enumerate(tokens):
        # print(tokens)
        word = token_info[2] if use_lemma else token_info[1]
        word = word.lower() if lowercase else word
        if replace_num and any(c.isdigit() for c in word):
            word = re.sub('[.|,]', '', word.lstrip('-'))
            if word.isdigit():
                word = NUM_WORD
        words.append(word)
        node = DepNode(token_idx)
        nodes.append(node)
    # build the tree structure
    root_node = None
    for token_idx, token_info in enumerate(tokens):
        parent_idx = int(token_info[5]) - 1
        # the root node
        if parent_idx == -1 and root_node is None:
            root_node = nodes[token_idx]
        # if there are multiple sentences, change the other root nodes as the children of the first
        elif parent_idx == -1:
            print('One Sentence has multiple root')
            print(conll_str.split('\n'))
            root_node.add_child(nodes[token_idx])
        else:
            nodes[parent_idx].add_child(nodes[token_idx])
    tree = DepTree()
    tree.assign_root(root_node)
    return words, tree


if __name__ == '__main__':
    raw_dir = '/home/yizhong/Workspace/Discourse/data/pdtb_v2/pipe'
    converted_dir = '/home/yizhong/Workspace/Discourse/data/pdtb_v2/converted'
    if not os.path.exists(converted_dir):
        os.mkdir(converted_dir)
    sections = os.listdir(raw_dir)
    core_nlp = StanfordCoreNLP('http://localhost:9000')
    annotate = lambda x: core_nlp.annotate(x, properties={
        'annotators': 'tokenize,ssplit,pos,lemma,parse,depparse',
        'outputFormat': 'json',
        'ssplit.isOneSentence': True
    })
    for section in sections:
        raw_sec_dir = os.path.join(raw_dir, section)
        if not os.path.isdir(raw_sec_dir):
            continue
        converted_sec_dir = os.path.join(converted_dir, section)
        if not os.path.exists(converted_sec_dir):
            os.mkdir(converted_sec_dir)
        for file in os.listdir(raw_sec_dir):
            convert_one_file(os.path.join(raw_sec_dir, file), converted_sec_dir, annotate)
    print('Totally, {} instances are converted.'.format(instance_cnt))

