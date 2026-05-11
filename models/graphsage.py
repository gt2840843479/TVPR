import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph.defense import GCN
import math


class SAGEConvolution(Module):
    """GraphSAGE convolutional layer with mean aggregation."""
    
    def __init__(self, in_features, out_features, with_bias=True):
        super(SAGEConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_neigh = Parameter(torch.FloatTensor(in_features, out_features))
        
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.weight_neigh.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input, adj):
        neigh_agg = torch.spmm(adj, input)
        if input.data.is_sparse:
            self_emb = torch.spmm(input, self.weight)
            neigh_emb = torch.spmm(neigh_agg, self.weight_neigh)
        else:
            self_emb = torch.mm(input, self.weight)
            neigh_emb = torch.mm(neigh_agg, self.weight_neigh)
        output = self_emb + neigh_emb
        
        if self.bias is not None:
            output += self.bias
        
        return output
    
    def __repr__(self):
        return f'{self.__class__.__name__} ({self.in_features} -> {self.out_features})'


class GraphSAGE(GCN): 
    """2-Layer GraphSAGE Network with mean aggregation."""

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,
                 with_relu=True, with_bias=True, device=None):
        super(GraphSAGE, self).__init__(nfeat, nhid, nclass, dropout, lr, weight_decay, with_relu, with_bias, device)
        
        self.gc1 = SAGEConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc2 = SAGEConvolution(nhid, nclass, with_bias=with_bias)
