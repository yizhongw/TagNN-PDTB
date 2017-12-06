import torch
import torch.nn as nn
from torch.autograd import Variable


class ChildSumTreeLSTM(nn.Module):
    def __init__(self, word_vocab_size, word_embed_dim, hidden_size, use_cuda):
        super(ChildSumTreeLSTM, self).__init__()
        self.use_cuda = use_cuda
        self.embed_dim = word_embed_dim
        self.hidden_size = hidden_size

        self.word_embed_func = nn.Embedding(word_vocab_size, word_embed_dim)
        self.ix = nn.Linear(self.embed_dim, self.hidden_size)
        self.ih = nn.Linear(self.hidden_size, self.hidden_size)
        self.fx = nn.Linear(self.embed_dim, self.hidden_size)
        self.fh = nn.Linear(self.hidden_size, self.hidden_size)
        self.ox = nn.Linear(self.embed_dim, self.hidden_size)
        self.oh = nn.Linear(self.hidden_size, self.hidden_size)
        self.ux = nn.Linear(self.embed_dim, self.hidden_size)
        self.uh = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, root_node, inputs):
        embs = torch.unsqueeze(self.word_embed_func(inputs), 1)
        outputs = []
        final_state = self.recursive_forward(root_node, embs, outputs)
        outputs = torch.cat(outputs, 0)
        return outputs, final_state

    def recursive_forward(self, node, embeds, outputs):
        child_c, child_h = None, None
        if len(node.children) == 0:
            child_c = Variable(torch.zeros(1, self.hidden_size))
            child_h = Variable(torch.zeros(1, self.hidden_size))
            if self.use_cuda:
                child_c, child_h = child_c.cuda(), child_h.cuda()
        else:
            for idx in range(node.children_num):
                c, h = self.recursive_forward(node.children[idx], embeds, outputs)
                if child_c is None and child_h is None:
                    child_c, child_h = c, h
                else:
                    child_c = torch.cat([child_c, c], 0)
                    child_h = torch.cat([child_h, h], 0)
        node_state = self.node_forward(embeds[node.idx], child_c, child_h)
        # store the hidden state of every node to outputs
        outputs.append(node_state[1])
        return node_state

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, 0)
        i = torch.sigmoid(self.ix(inputs) + self.ih(child_h_sum))
        o = torch.sigmoid(self.ox(inputs) + self.oh(child_h_sum))
        fi = self.fx(inputs)
        f = torch.cat([self.fh(child_hi) + fi for child_hi in torch.unsqueeze(child_h, 1)], 0)
        f = torch.sigmoid(f)
        fc = torch.mul(f, child_c)
        u = torch.tanh(self.ux(inputs) + self.uh(child_h_sum))
        c = torch.mul(i, u) + torch.sum(fc, 0)
        h = torch.mul(o, torch.tanh(c))
        return c, h
