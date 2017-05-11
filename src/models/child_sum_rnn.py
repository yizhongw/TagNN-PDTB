import torch
import torch.nn as nn
from torch.autograd import Variable


class ChildSumTreeLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, use_cuda):
        super(ChildSumTreeLSTM, self).__init__()
        self.use_cuda = use_cuda
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.ix = nn.Linear(self.embed_dim, self.hidden_size)
        self.ih = nn.Linear(self.hidden_size, self.hidden_size)
        self.fx = nn.Linear(self.embed_dim, self.hidden_size)
        self.fh = nn.Linear(self.hidden_size, self.hidden_size)
        self.ox = nn.Linear(self.embed_dim, self.hidden_size)
        self.oh = nn.Linear(self.hidden_size, self.hidden_size)
        self.ux = nn.Linear(self.embed_dim, self.hidden_size)
        self.uh = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, root_node, inputs):
        embs = torch.unsqueeze(self.emb(inputs), 1)
        outputs = []
        final_state = self.recursive_forward(root_node, embs, outputs)
        outputs = torch.cat(outputs, 0)
        return outputs, final_state

    def recursive_forward(self, node, embeds, outputs):
        for idx in range(node.children_num):
            _ = self.recursive_forward(node.children[idx], embeds, outputs)
        child_c, child_h = self.get_child_states(node)
        node.state = self.node_forward(embeds[node.idx], child_c, child_h)
        # store the hidden state of every node to outputs
        outputs.append(node.state[1])
        return node.state

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(torch.squeeze(child_h, 1), 0)
        i = torch.sigmoid(self.ix(inputs) + self.ih(child_h_sum))
        o = torch.sigmoid(self.ox(inputs) + self.oh(child_h_sum))
        fi = self.fx(inputs)
        f = torch.cat([self.fh(child_hi) + fi for child_hi in child_h], 0)
        f = torch.sigmoid(f)
        fc = torch.mul(f, torch.squeeze(child_c, 1))
        u = torch.tanh(self.ux(inputs) + self.uh(child_h_sum))
        c = torch.mul(i, u) + torch.sum(fc, 0)
        h = torch.mul(o, torch.tanh(c))
        return c, h

    def get_child_states(self, node):
        if len(node.children) == 0:
            child_c = Variable(torch.zeros(1, 1, self.hidden_size))
            child_h = Variable(torch.zeros(1, 1, self.hidden_size))
            if self.use_cuda:
                child_c, child_h = child_c.cuda(), child_h.cuda()
        else:
            child_c = Variable(torch.Tensor(node.children_num, 1, self.hidden_size))
            child_h = Variable(torch.Tensor(node.children_num, 1, self.hidden_size))
            if self.use_cuda:
                child_c, child_h = child_c.cuda(), child_h.cuda()
            for idx in range(node.children_num):
                child_c[idx], child_h[idx] = node.children[idx].state
        return child_c, child_h
