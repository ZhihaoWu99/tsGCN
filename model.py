import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, device, bias, activation=F.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.device = device
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.activation = activation
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, lp):
        x = torch.mm(inputs, self.weight)
        output = torch.mm(lp, x)
        if self.activation:
            output = self.activation(output)
        return output


class GCN(nn.Module):
    def __init__(self, hidden_dims, dropout, bias, device):
        super(GCN, self).__init__()
        self.device = device
        self.gc = nn.ModuleList()
        for i in range(len(hidden_dims)-2):
            self.gc.append(GraphConvolution(hidden_dims[i], hidden_dims[i+1], self.device, bias))
        self.gc.append(GraphConvolution(hidden_dims[-2], hidden_dims[-1], self.device, bias, None))
        self.dropout = dropout

    def forward(self, x, lp):
        emb = x
        for gc in self.gc:
            emb = gc(emb, lp)
        if self.dropout != 0:
            emb = F.dropout(emb, self.dropout, training=self.training)
        return emb