import torch
import torch.nn as nn
from torch.autograd import Variable


class BinaryTreeLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, use_cuda):
        super(BinaryTreeLSTM, self).__init__()
        self.use_cuda = use_cuda
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.ix = nn.Linear(self.embed_dim, self.hidden_size)
        self.il = nn.Linear(self.hidden_size, self.hidden_size)
        self.ir = nn.Linear(self.hidden_size, self.hidden_size)
        self.fx = nn.Linear(self.embed_dim, self.hidden_size)
        self.fl = nn.Linear(self.hidden_size, self.hidden_size)
        self.fr = nn.Linear(self.hidden_size, self.hidden_size)
        self.ox = nn.Linear(self.embed_dim, self.hidden_size)
        self.ol = nn.Linear(self.hidden_size, self.hidden_size)
        self.or_ = nn.Linear(self.hidden_size, self.hidden_size)
        self.ux = nn.Linear(self.embed_dim, self.hidden_size)
        self.ul = nn.Linear(self.hidden_size, self.hidden_size)
        self.ur = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, root_node, inputs):
        embs = torch.unsqueeze(self.emb(inputs), 1)
        outputs = []
        final_state = self.recursive_forward(root_node.children[0], embs, outputs)
        outputs = torch.cat(outputs, 0)
        return outputs, final_state

    def recursive_forward(self, node, embeds, outputs):
        for idx in range(len(node.children)):
            _ = self.recursive_forward(node.children[idx], embeds, outputs)
        left_states, right_states = self.get_child_states(node)
        if node.idx is not None:
            node.state = self.node_forward(embeds[node.idx], left_states, right_states)
        else:
            embed = Variable(torch.zeros(1, self.embed_dim))
            if self.use_cuda:
                embed = embed.cuda()
            node.state = self.node_forward(embed, left_states, right_states)
        outputs.append(node.state[1])
        return node.state

    def node_forward(self, inputs, left_states, right_states):
        left_c, left_h = left_states
        right_c, right_h = right_states
        i = torch.sigmoid(self.ix(inputs) + self.il(left_h) + self.ir(right_h))
        f = torch.sigmoid(self.fx(inputs) + self.fl(left_h) + self.fr(right_h))
        o = torch.sigmoid(self.ox(inputs) + self.ol(left_h) + self.or_(right_h))
        u = torch.tanh(self.ux(inputs) + self.ul(left_h) + self.ur(right_h))
        c = torch.mul(i, u) + torch.mul(f, left_c) + torch.mul(f, right_c)
        h = torch.mul(o, torch.tanh(c))
        return c, h

    def get_child_states(self, node):
        if len(node.children) == 0:
            left_c = Variable(torch.zeros(1, self.hidden_size))
            left_h = Variable(torch.zeros(1, self.hidden_size))
            right_c = Variable(torch.zeros(1, self.hidden_size))
            right_h = Variable(torch.zeros(1, self.hidden_size))
            if self.use_cuda:
                left_c, left_h = left_c.cuda(), left_h.cuda()
                right_c, right_h = right_c.cuda(), right_h.cuda()
            return (left_c, left_h), (right_c, right_h)
        else:
            try:
                assert len(node.children) == 2
            except Exception as e:
                print(len(node.children))
                print(node.to_string())
                raise e
            return node.children[0].state, node.children[1].state
