import torch
import torch.nn as nn
from torch.autograd import Variable


class BinaryTreeLSTM(nn.Module):
    def __init__(self, word_vocab_size, word_embed_dim, hidden_size, use_cuda):
        super(BinaryTreeLSTM, self).__init__()
        self.use_cuda = use_cuda
        self.word_embed_dim = word_embed_dim
        self.hidden_size = hidden_size

        self.word_embed_func = nn.Embedding(word_vocab_size, word_embed_dim)
        self.ig = nn.Linear(self.word_embed_dim + 2 * self.hidden_size, self.hidden_size)
        self.fg = nn.Linear(self.word_embed_dim + 2 * self.hidden_size, self.hidden_size)
        # self.fgi = nn.Linear(self.word_embed_dim, self.hidden_size)
        # self.fgl = nn.Linear(2 * self.hidden_size, self.hidden_size)
        # self.fgr = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.og = nn.Linear(self.word_embed_dim + 2 * self.hidden_size, self.hidden_size)
        self.u = nn.Linear(self.word_embed_dim + 2 * self.hidden_size, self.hidden_size)

    def forward(self, root_node, inputs):
        outputs = []
        final_state = self.recursive_forward(root_node.children[0], inputs, outputs)
        outputs = torch.cat(outputs, 0)
        return outputs, final_state

    def recursive_forward(self, node, inputs, outputs):
        # get states from children
        child_states = []
        if len(node.children) == 0:
            left_c = Variable(torch.zeros(1, self.hidden_size))
            left_h = Variable(torch.zeros(1, self.hidden_size))
            right_c = Variable(torch.zeros(1, self.hidden_size))
            right_h = Variable(torch.zeros(1, self.hidden_size))
            if self.use_cuda:
                left_c, left_h = left_c.cuda(), left_h.cuda()
                right_c, right_h = right_c.cuda(), right_h.cuda()
            child_states.append((left_c, left_h))
            child_states.append((right_c, right_h))
        else:
            assert len(node.children) <= 2
            for idx in range(len(node.children)):
                child_state = self.recursive_forward(node.children[idx], inputs, outputs)
                child_states.append(child_state)
        # calculate the state of current node
        node_state = self.node_forward(node, child_states)
        outputs.append(node_state[1])
        return node_state

    def node_forward(self, node, child_states):
        if node.idx is not None:
            node_word = node.word
            if self.use_cuda:
                node_word = node_word.cuda()
            word_embed = self.word_embed_func(node_word)
        else:
            word_embed = Variable(torch.zeros(1, self.word_embed_dim))
            if self.use_cuda:
                word_embed = word_embed.cuda()
        if len(child_states) == 1:
            return child_states[0]
        else:
            left_c, left_h = child_states[0]
            right_c, right_h = child_states[1]
            i = torch.sigmoid(self.ig(torch.cat([word_embed, left_h, right_h], 1)))
            f = torch.sigmoid(self.fg(torch.cat([word_embed, left_h, right_h], 1)))
            # fl = torch.sigmoid(self.fgi(word_embed) + self.fgl(torch.cat([left_h, right_h], 1)))
            # fr = torch.sigmoid(self.fgi(word_embed) + self.fgr(torch.cat([left_h, right_h], 1)))
            o = torch.sigmoid(self.og(torch.cat([word_embed, left_h, right_h], 1)))
            u = torch.tanh(self.u(torch.cat([word_embed, left_h, right_h], 1)))
            c = torch.mul(i, u) + torch.mul(f, left_c) + torch.mul(f, right_c)
            h = torch.mul(o, torch.tanh(c))
            return c, h


class LabeledBinaryTreeLSTM(nn.Module):
    def __init__(self, word_vocab_size, tag_vocab_size, word_embed_dim, tag_embed_dim, hidden_size, use_cuda):
        super(LabeledBinaryTreeLSTM, self).__init__()
        self.use_cuda = use_cuda
        self.word_embed_dim = word_embed_dim
        self.tag_embed_dim = tag_embed_dim
        self.hidden_size = hidden_size

        self.word_embed_func = nn.Embedding(word_vocab_size, word_embed_dim)
        self.tag_embed_func = nn.Embedding(tag_vocab_size, tag_embed_dim)
        self.ig = nn.Linear(self.word_embed_dim + self.tag_embed_dim + 2 * self.hidden_size, self.hidden_size)
        self.fg = nn.Linear(self.word_embed_dim + self.tag_embed_dim + 2 * self.hidden_size, self.hidden_size)
        # self.fgi = nn.Linear(self.word_embed_dim + self.tag_embed_dim, self.hidden_size)
        # self.fgl = nn.Linear(2 * self.hidden_size, self.hidden_size)
        # self.fgr = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.og = nn.Linear(self.word_embed_dim + self.tag_embed_dim + 2 * self.hidden_size, self.hidden_size)
        self.u = nn.Linear(self.word_embed_dim + 2 * self.hidden_size, self.hidden_size)

    def forward(self, root_node, inputs):
        outputs = []
        final_state = self.recursive_forward(root_node, inputs, outputs)
        outputs = torch.cat(outputs, 0)
        return outputs, final_state

    def recursive_forward(self, node, inputs, outputs):
        # get states from children
        child_states = []
        if len(node.children) == 0:
            left_c = Variable(torch.zeros(1, self.hidden_size))
            left_h = Variable(torch.zeros(1, self.hidden_size))
            right_c = Variable(torch.zeros(1, self.hidden_size))
            right_h = Variable(torch.zeros(1, self.hidden_size))
            if self.use_cuda:
                left_c, left_h = left_c.cuda(), left_h.cuda()
                right_c, right_h = right_c.cuda(), right_h.cuda()
            child_states.append((left_c, left_h))
            child_states.append((right_c, right_h))
        else:
            assert len(node.children) <= 2
            for idx in range(len(node.children)):
                child_state = self.recursive_forward(node.children[idx], inputs, outputs)
                child_states.append(child_state)
        # calculate the state of current node
        node_state = self.node_forward(node, child_states)
        outputs.append(node_state[1])
        return node_state

    def node_forward(self, node, child_states):
        if node.idx is not None:
            node_word = node.word
            if self.use_cuda:
                node_word = node_word.cuda()
            word_embed = self.word_embed_func(node_word)
        else:
            word_embed = Variable(torch.zeros(1, self.word_embed_dim))
            if self.use_cuda:
                word_embed = word_embed.cuda()
        node_tag = node.tag
        if self.use_cuda:
            node_tag = node_tag.cuda()
        tag_embed = self.tag_embed_func(node_tag)
        if len(child_states) == 1:
            return child_states[0]
        else:
            left_c, left_h = child_states[0]
            right_c, right_h = child_states[1]
            input = torch.cat([word_embed, tag_embed], 1)
            i = torch.sigmoid(self.ig(torch.cat([input, left_h, right_h], 1)))
            f = torch.sigmoid(self.fg(torch.cat([input, left_h, right_h], 1)))
            # fl = torch.sigmoid(self.fgi(input) + self.fgl(torch.cat([left_h, right_h], 1)))
            # fr = torch.sigmoid(self.fgi(input) + self.fgr(torch.cat([left_h, right_h], 1)))
            o = torch.sigmoid(self.og(torch.cat([input, left_h, right_h], 1)))
            u = torch.tanh(self.u(torch.cat([word_embed, left_h, right_h], 1)))
            c = torch.mul(i, u) + torch.mul(f, left_c) + torch.mul(f, right_c)
            h = torch.mul(o, torch.tanh(c))
            return c, h