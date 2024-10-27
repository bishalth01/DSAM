import torch
import torch.nn.functional as F
from torch.nn import Parameter
from net.brainmsgpassing import MyMessagePassing
from torch_geometric.utils import add_remaining_self_loops,softmax
import torch.nn.init as init
from torch_geometric.typing import (OptTensor)

from net.inits import uniform
import numpy as np


class MyGINConvWithMean(MyMessagePassing):
    def __init__(self, in_channels, out_channels, nn, normalize=False, bias=False, eps=0,
                 **kwargs):
        super(MyGINConvWithMean, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.nn = nn

        # initialize epsilon to zero
        self.eps = torch.nn.Parameter(torch.zeros(2+1))
        init.normal_(self.eps)
        #self.eps = eps
        self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        #self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        self.mlp = torch.nn.Sequential(torch.nn.Linear(self.out_channels, self.out_channels * 2),
                                        torch.nn.BatchNorm1d(self.out_channels * 2),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(self.out_channels * 2, self.out_channels))

        # self.mlp = torch.nn.Sequential(torch.nn.Linear(self.out_channels, self.out_channels),
        #                                 torch.nn.BatchNorm1d(self.out_channels),
        #                                 torch.nn.ReLU()
        #                                 )

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
#        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)
        self.eps.data.fill_(0.0)

    def forward(self, x, edge_index, edge_weight=None, pseudo= None, size=None):
        """"""
        edge_weight = edge_weight.squeeze()
        # if size is None and torch.is_tensor(x):
        #     edge_index, edge_weight = add_remaining_self_loops(
        #         edge_index, edge_weight, 1, x.size(0))

        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
        if torch.is_tensor(x):
            x = torch.matmul(x.unsqueeze(1), weight).squeeze(1)  #W_i*h_i
        else:
            x = (None if x[0] is None else torch.matmul(x[0].unsqueeze(1), weight).squeeze(1),
                 None if x[1] is None else torch.matmul(x[1].unsqueeze(1), weight).squeeze(1))
            
        #out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, size=size, x=x, edge_weight=edge_weight))
        #out = self.propagate(edge_index, size=size, x=x, edge_weight=edge_weight)

        aggregation = self.propagate(edge_index, size=size, x=x, edge_weight=edge_weight)

        #x_gin = F.relu(self.mlp((1+self.eps) * x) + x_rgcn)
        x_gin = self.mlp(((1+self.eps) * x) + aggregation)

        #out = x_rgcn + x_gin
        out = x_gin

        return out
    
    # def gin_layer(self, x, edge_index, edge_weight):
    #     x = self.gin_conv(x, edge_index, edge_weight)
    #     x = F.relu(x)
    #     x = F.dropout(x, p=self.dropout, training=self.training)
    #     return x

    def message(self, edge_index_i, size_i, x_j, edge_weight, ptr: OptTensor):
        edge_weight = softmax(edge_weight, edge_index_i, ptr, size_i)
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    # def update(self, aggr_out):
    #     if self.bias is not None:
    #         aggr_out = aggr_out + self.bias
    #     if self.normalize:
    #         aggr_out = F.normalize(aggr_out, p=2, dim=-1)
    #     return aggr_out

    def update(self, aggr_out):
        # learnable weights 
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
            
        #aggr_out = self.nn((1 + self.eps) * aggr_out)
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)

        return aggr_out
    
    
    # def update(self, aggr_out, x):
    #     learnable_weights = self.bn(self.lin(x))
    #     out = aggr_out + learnable_weights
    #     out = torch.nn.functional.relu(out)
    #     return out
    
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

