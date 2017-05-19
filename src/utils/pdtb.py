#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: yizhong
# created_at: 17-4-27 下午7:07
import os
import pickle
from copy import deepcopy
from torch.utils.data import Dataset


class PDTBDataSet(Dataset):
    def __init__(self, section_dirs, tree_type='dependency', level=1, multiple_labels=False):
        super(PDTBDataSet, self).__init__()
        self.tree_type = tree_type
        self.level = level
        self.multiple_labels = multiple_labels
        self.instances = self._load_implicit_instances(section_dirs)
        self.label_map = self._build_label_map()
        self.size = len(self.instances)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        inst = self.instances[index]
        lsent, rsent = inst.arg1_words, inst.arg2_words
        if self.tree_type == 'dependency':
            ltree, rtree = inst.arg1_dep_tree, inst.arg2_dep_tree
        elif self.tree_type == 'constituency':
            ltree, rtree = inst.arg1_const_tree, inst.arg2_const_tree
        else:
            raise NotImplementedError('Unsupported tree type: {}'.format(self.tree_type))
        if self.multiple_labels:
            label = [self.label_map[sem] for sem in inst.sems]
        else:
            label = self.label_map[inst.sem]
        return ltree, lsent, rtree, rsent, label

    def _load_implicit_instances(self, section_dirs):
        instances = []
        for section_dir in section_dirs:
            for file in os.listdir(section_dir):
                if not file.endswith('pickle'):
                    continue
                with open(os.path.join(section_dir, file), 'rb') as fin:
                    inst = pickle.load(fin)
                    if inst.type != 'Implicit':
                        continue
                    if self.multiple_labels:
                        sems = set()
                        for sem in inst.sems:
                            sem = get_subtype(sem, self.level)
                            if sem is not None:
                                sems.add(sem)
                        inst.sems = sems
                        inst.sem = get_subtype(inst.sem, self.level)
                        instances.append(inst)
                    else:
                        for sem in inst.sems:
                            sem = get_subtype(sem, self.level)
                            if sem is not None:
                                new_inst = deepcopy(inst)
                                new_inst.sem = sem
                                instances.append(new_inst)
        return instances

    def _build_label_map(self):
        if self.multiple_labels:
            labels = set([sem for inst in self.instances for sem in inst.sems])
        else:
            labels = set([inst.sem for inst in self.instances])
        label_map = {label: idx for idx, label in enumerate(sorted(labels))}
        return label_map

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


if __name__ == '__main__':
    dataset = PDTBDataSet(['/home/yizhong/Workspace/Discourse/data/pdtb_v2/converted/00'])
    print(dataset.get_all_words())
