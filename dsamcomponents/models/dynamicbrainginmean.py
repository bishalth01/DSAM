import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
import math
from torch_sparse import spspmm
import numpy as np
#from Pooling.DiffPoolLayer import DiffPoolLayer
import torch_geometric.utils as pyg_utils
from dsamcomponents.models.braingraphconv import MyNNConv
from dsamcomponents.models.bgnngingraphconv import MyGINConv
#from net.graphisographconv import MyGINConv
from torch_scatter import scatter_mean, scatter_add
from einops import rearrange, repeat
#from net.MyPNAConv import PNAConv
from torch_geometric.nn.aggr.gmt import GraphMultisetTransformer


##########################################################################################################################
class CustomNetworkGINMean(torch.nn.Module):
    def __init__(self, indim, ratio, nclass, n_hidden_layers, n_fc_layers, k, fc_dropout=0.2,R=100):
        '''

        :param indim: (int) node feature dimension
        :param ratio: (float) pooling ratio in (0,1)
        :param nclass: (int)  number of classes
        :param k: (int) number of communities
        :param R: (int) number of ROIs
        '''
        super(CustomNetworkGINMean, self).__init__()
        self.n_hidden_layers = n_hidden_layers
        self.n_fc_layers = n_fc_layers
        self.indim = indim
        self.fc_dropout=fc_dropout

        self.k = k
        self.R = R

        self.allnns = nn.ModuleList()
        self.allconvs = nn.ModuleList()
        self.allpools = nn.ModuleList()

        #Fully Connected Layers
        self.allfcs = nn.ModuleList()
        self.batchnorms= nn.ModuleList()

        #Graph Convolution and Pooling

        for i in range(len(n_hidden_layers)):
            if(i==0):
                self.allnns.append(nn.Sequential(nn.Linear(self.R, self.k, bias=False), nn.ReLU(), nn.Linear(self.k, n_hidden_layers[i] * self.indim)))
                self.allconvs.append(MyGINConvWithMean(self.indim, n_hidden_layers[i], self.allnns[i], normalize=False))
                self.allpools.append(TopKPooling(n_hidden_layers[i], ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid))
            else:
                self.allnns.append(nn.Sequential(nn.Linear(self.R, self.k, bias=False), nn.ReLU(), nn.Linear(self.k, n_hidden_layers[i] * n_hidden_layers[i-1])))
                self.allconvs.append(MyGINConvWithMean(n_hidden_layers[i-1], n_hidden_layers[i], self.allnns[i], normalize=False))
                self.allpools.append(TopKPooling(n_hidden_layers[i], ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid))
        
        #Fully Connected Layers and Batch Norms

        for i in range(len(n_fc_layers)):
            if i==0:
                self.allfcs.append(torch.nn.Linear(np.sum(n_hidden_layers)*2, n_fc_layers[i]))
                self.batchnorms.append(torch.nn.BatchNorm1d(n_fc_layers[i]))
            else:
                self.allfcs.append(torch.nn.Linear(n_fc_layers[i-1], n_fc_layers[i]))
                self.batchnorms.append(torch.nn.BatchNorm1d(n_fc_layers[i]))

        #self.finallayer = torch.nn.Linear(n_fc_layers[len(n_fc_layers)-1], nclass)
        self.finallayer = torch.nn.Linear(n_fc_layers[len(n_fc_layers)-1], 2)




    def forward(self, x, edge_index, batch, edge_attr, pos):

        #Graph Convolution Part

        all_outputs = []
        scores=[]

        for i in range(len(self.n_hidden_layers)):  #32,32

            x = self.allconvs[i](x, edge_index, edge_attr, pos)

            
            x, edge_index, edge_attr, batch, perm, score = self.allpools[i](x, edge_index, edge_attr, batch)
            pos = pos[perm]
            all_outputs.append(torch.cat([gmp(x, batch), gap(x, batch)], dim=1))
            scores.append(score)

            edge_attr = edge_attr.squeeze()
            edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))

        #Fully Connected Layer Part

        for i in range(len(self.n_fc_layers)):
            if i==0:
                x = torch.cat(all_outputs, dim=1)
                x = self.batchnorms[i](F.relu(self.allfcs[i](x)))
                x = F.dropout(x, p=self.fc_dropout, training=self.training)
            else:
                x = self.batchnorms[i](F.relu(self.allfcs[i](x)))
                x= F.dropout(x, p=self.fc_dropout, training=self.training)
        
        #x = torch.sigmoid(self.finallayer(x))
        #x= self.finallayer(x)
        x = F.log_softmax(self.finallayer(x), dim=-1)

        return x, self.allpools, scores, edge_attr

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

