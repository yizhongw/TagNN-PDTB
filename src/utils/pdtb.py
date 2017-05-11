#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: yizhong
# created_at: 17-4-27 下午7:07
import os
import pickle
import random
from torch.utils.data import Dataset


class PDTBDataSet(Dataset):
    def __init__(self, section_dirs, tree_type='dependency', level=1):
        super(PDTBDataSet, self).__init__()
        self.tree_type = tree_type
        self.level = level
        self.instances = self._load_implicit_instances(section_dirs)
        self.label_map = self._build_label_map()
        self.instances_by_label_id = self._classify_instances_by_label_id()
        self.size = len(self.instances)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        inst = self.instances[index]
        if self.tree_type == 'dependency':
            lsent, ltree, rsent, rtree, relation = inst.arg1_words, inst.arg1_dep_tree, inst.arg2_words, inst.arg2_dep_tree, inst.sem
        elif self.tree_type == 'constituency':
            lsent, ltree, rsent, rtree, relation = inst.arg1_words, inst.arg1_const_tree, inst.arg2_words, inst.arg2_const_tree, inst.sem
        else:
            raise NotImplementedError('Unsupported tree type: {}'.format(self.tree_type))
        return ltree, lsent, rtree, rsent, self.label_map[relation]

    def get_item_by_label(self, label_id):
        inst = random.choice(self.instances_by_label_id[label_id])
        if self.tree_type == 'dependency':
            lsent, ltree, rsent, rtree, relation = inst.arg1_words, inst.arg1_dep_tree, inst.arg2_words, inst.arg2_dep_tree, inst.sem
        elif self.tree_type == 'constituency':
            lsent, ltree, rsent, rtree, relation = inst.arg1_words, inst.arg1_const_tree, inst.arg2_words, inst.arg2_const_tree, inst.sem
        else:
            raise NotImplementedError('Unsupported tree type: {}'.format(self.tree_type))
        return ltree, lsent, rtree, rsent, self.label_map[relation]

    def _load_implicit_instances(self, section_dirs):
        instances = []
        for section_dir in section_dirs:
            for file in os.listdir(section_dir):
                if not file.endswith('pickle'):
                    continue
                with open(os.path.join(section_dir, file), 'rb') as fin:
                    inst = pickle.load(fin)
                    if inst.type == 'Implicit' and len(inst.sems) == 1:
                        inst.sem = get_subtype(inst.sem, self.level)
                        if self.level == 2 and inst.sem in ['Comparison.Pragmatic contrast', 'Contingency',
                                                            'Contingency.Condition',
                                                            'Contingency.Pragmatic condition', 'Expansion.Exception']:
                            continue
                        instances.append(inst)
        return instances

    def _build_label_map(self):
        labels = set([inst.sem for inst in self.instances])
        label_map = {label: idx for idx, label in enumerate(sorted(labels))}
        return label_map

    def _classify_instances_by_label_id(self):
        instances_by_label_id = {}
        for inst in self.instances:
            label_id = self.label_map[inst.sem]
            if label_id in instances_by_label_id:
                instances_by_label_id[label_id].append(inst)
            else:
                instances_by_label_id[label_id] = [inst]
        return instances_by_label_id

    def get_all_words(self):
        words = []
        for inst in self.instances:
            words.extend(inst.arg1_words)
            words.extend(inst.arg2_words)
        return words

    def consistent_with(self, dataset_2):
        '''
        check whether two dataset have the same label map
        :param dataset_2: PDTBDataSet
        :return: Boolean
        '''
        all_labels = set(self.label_map.keys()) | set(dataset_2.label_map.keys())
        if not len(all_labels) == len(self.label_map.keys()) == len(dataset_2.label_map.keys()):
            return False
        for label in all_labels:
            if self.label_map[label] != dataset_2.label_map[label]:
                return False
        return True


class PDTBInstance:
    def __init__(self, columns, level):
        self.type = columns[0]
        self.section = columns[1]
        self.file = columns[2]
        self.arg1 = columns[24]
        self.arg2 = columns[34]
        self.connective = columns[5] # only for Explicit and AltLex
        self.connective_start = int(columns[3].split('..')[0]) if self.connective != '' else -1
        self.conn1 = columns[9] # only for Implicit
        self.conn2 = columns[10] # only for Implicit
        self.conn1_sem1 = get_subtype(columns[11], level) # only for Explicit, Implicit and AltLex
        self.conn1_sem2 = get_subtype(columns[12], level) # only for Explicit, Implicit and AltLex
        self.conn2_sem1 = get_subtype(columns[13], level) # only for Implicit
        self.conn2_sem2 = get_subtype(columns[14], level) # only for Implicit
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
    if sem == '':
        return ''
    else:
        return '.'.join(sem.split('.')[:level])


if __name__ == '__main__':
    dataset = PDTBDataSet(['/home/yizhong/Workspace/Discourse/data/pdtb_v2/converted/00'])
    print(dataset.get_all_words())
