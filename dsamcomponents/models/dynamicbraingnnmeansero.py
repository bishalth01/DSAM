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
import torch_geometric.utils as pyg_utils
from net.braingraphconv import MyNNConv
from net.ginconvwithmeanagg import MyGINConvWithMean
#from net.graphisographconv import MyGINConv
from torch_scatter import scatter_mean, scatter_add
from einops import rearrange, repeat
#from net.MyPNAConv import PNAConv
from torch_geometric.nn.aggr.gmt import GraphMultisetTransformer

##########################################################################################################################
class CustomNetworkGINSeroMean(torch.nn.Module):
    def __init__(self, indim, ratio, nclass, n_hidden_layers, n_fc_layers, k, fc_dropout=0.2,R=100):
        '''

        :param indim: (int) node feature dimension
        :param ratio: (float) pooling ratio in (0,1)
        :param nclass: (int)  number of classes
        :param k: (int) number of communities
        :param R: (int) number of ROIs
        '''
        super(CustomNetworkGINSeroMean, self).__init__()
        self.n_hidden_layers = n_hidden_layers
        self.n_fc_layers = n_fc_layers
        self.indim = indim
        self.ratio = ratio
        self.fc_dropout = fc_dropout
        #self.reg = reg

        self.k = k
        self.R = R

        self.allnns = nn.ModuleList()
        self.allconvs = nn.ModuleList()
        self.allpools = nn.ModuleList()
        self.gmts = nn.ModuleList()

        #Fully Connected Layers
        self.allfcs = nn.ModuleList()
        self.batchnorms= nn.ModuleList()

        self.pnaconvs = nn.ModuleList()
        self.seros = nn.ModuleList()

        #Graph Convolution and Pooling
        

        for i in range(len(n_hidden_layers)):
            if(i==0):
                self.allnns.append(nn.Sequential(nn.Linear(self.R, self.k, bias=False), nn.ReLU(), nn.Linear(self.k, n_hidden_layers[i] * self.indim)))
                self.allconvs.append(MyGINConvWithMean(self.indim, n_hidden_layers[i], self.allnns[i], normalize=False))
                self.allpools.append(TopKPooling(n_hidden_layers[i], ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid))
                #self.gmts.append(GraphMultisetTransformer(n_hidden_layers[i], 128, n_hidden_layers[i], num_nodes=R))
                self.seros.append(ModuleSERO(output_dim=n_hidden_layers[i], hidden_dim=n_hidden_layers[i], dropout=0.1, upscale=1.0))


                #self.pnaconvs.append(MyPNAConv())
            else:
                self.allnns.append(nn.Sequential(nn.Linear(self.R, self.k, bias=False), nn.ReLU(), nn.Linear(self.k, n_hidden_layers[i] * n_hidden_layers[i-1])))
                self.allconvs.append(MyGINConvWithMean(n_hidden_layers[i-1], n_hidden_layers[i], self.allnns[i], normalize=False))
                self.allpools.append(TopKPooling(n_hidden_layers[i], ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid))
                #self.gmts.append(GraphMultisetTransformer(n_hidden_layers[i] , 128, n_hidden_layers[i] , num_nodes=R))
                self.seros.append(ModuleSERO(output_dim=n_hidden_layers[i], hidden_dim=n_hidden_layers[i], dropout=0.1, upscale=1.0))
        
       

        #features_remaining = self.n_final_nodes.reshape(64,num_remaining_features).shape[-1]
        #self.diff_pool = DiffPoolLayer(self.R, self.n_hidden_layers[-1], aggr='max')
        #self.sero = ModuleSERO(input_dim=32, hidden_dim=64, dropout=0.1, upscale=1.0)


        for i in range(len(n_fc_layers)):
            if i==0:
                final_layer_output = sum(2*int(x) for x in n_hidden_layers)
                final_conv_layer = n_hidden_layers[-1]
                self.allfcs.append(torch.nn.Linear((np.sum(n_hidden_layers) ), n_fc_layers[i]))
                #self.allfcs.append(torch.nn.Linear(96, n_fc_layers[i]))
                #self.allfcs.append(torch.nn.Linear(7 * final_conv_layer, n_fc_layers[i]))
                #self.allfcs.append(torch.nn.Linear((np.sum(n_hidden_layers[-1]) + np.sum(n_hidden_layers)), n_fc_layers[i]))
                #self.allfcs.append(torch.nn.Linear(final_layer_output + (self.num_remaining_features*256), n_fc_layers[i]))
                self.batchnorms.append(torch.nn.BatchNorm1d(n_fc_layers[i]))
            else:
                self.allfcs.append(torch.nn.Linear(n_fc_layers[i-1], n_fc_layers[i]))
                self.batchnorms.append(torch.nn.BatchNorm1d(n_fc_layers[i]))
        
        

        self.finallayer = torch.nn.Linear(n_fc_layers[len(n_fc_layers)-1], nclass)




    def forward(self, x, edge_index, batch, edge_attr, pos):

        #Graph Convolution Part

        all_outputs = []
        all_outputs_gin =[]
        individual_layer_output=[]
        scores=[]
        gmts=[]
        seros=[]
        batch_size = x.shape[0]//self.R

        for i in range(len(self.n_hidden_layers)):

            x = self.allconvs[i](x, edge_index, edge_attr, pos)
            #x_gmt  = self.gmts[i](x, index= batch,edge_index = edge_index)
            
            x, edge_index, edge_attr, batch, perm, score = self.allpools[i](x, edge_index, edge_attr, batch)
            pos = pos[perm]
            #x_gmt  = self.gmts[i](x, index= batch,edge_index = edge_index)
            sero_output,_ = self.seros[i](x)
            all_outputs.append(torch.cat([gmp(x, batch), gap(x, batch)], dim=1))
            #all_outputs.append(torch.cat([gmp(x_gmt, batch), gap(x_gmt, batch)], dim=1))

            #all_outputs_gin.append(self.global_add_pool(x,batch))
            scores.append(score)


            edge_attr = edge_attr.squeeze()
            edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))
            # individual_layer_output.append(x)
            #gmts.append(x_gmt)
            seros.append(sero_output)


        
        #Fully Connected Layer Part

        #x_importantnodes, indices = torch.topk(x, k= self.num_remaining_features * 64, dim=0)
        #x = x[indices]
        #final_x = x_importantnodes.reshape(64, -1)

        for i in range(len(self.n_fc_layers)):
            if i==0:
                #adj_tmp = pyg_utils.to_dense_adj(edge_index, batch, edge_attr=edge_attr)
                # if edge_attr is not None: # Because edge_attr only has 1 feature per edge
                #     adj_tmp = adj_tmp[:, :, :, 0]
                #x_tmp, batch_mask = pyg_utils.to_dense_batch(x, batch)
                #x, l1, l2 = self.diff_pool(x_tmp, adj_tmp, batch_mask )
                #x = torch.concat(x, dim=1)
                #x = self.select_orthonormal_features(x, 7, batch_size)
                #x = torch.randn(128, 1792).to(device='cuda')

                #x = torch.randn(128, 512).to(device='cuda')

                #x = torch.concat(all_outputs, dim=1)
                x = torch.concat(seros, dim=1)
                #x = self.select_orthonormal_features(x, 7, batch_size)
                #x = torch.concat(gmts, dim=1)


                #x = torch.concat([gmts[-1], all_outputs[-1]], dim=1)
                #x = gmts[-1]
                #x = torch.concat(all_outputs_gin, dim=1)
                #x_pool = torch.concat(all_outputs, dim=1)
                # x_gmt = self.reg * (torch.concat(gmts, dim=1))
                # x = torch.cat([x_pool, x_gmt], dim=1)
                x = self.batchnorms[i](F.relu(self.allfcs[i](x)))
                x = F.dropout(x, p=self.fc_dropout, training=self.training)
            else:
                x = self.batchnorms[i](F.relu(self.allfcs[i](x)))
                x= F.dropout(x, p=self.fc_dropout, training=self.training)
        
        x = torch.relu(self.finallayer(x))

        return x,self.allpools, scores


    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight


  
    def select_orthonormal_features(self, x, k, batch_size):

        _, feat_dim = x.shape
        num_nodes = _//batch_size
        node_features = x.reshape(batch_size, num_nodes, feat_dim)
        node_features = node_features / torch.norm(node_features, dim=2, keepdim=True)

        # compute covariance matrices using matrix multiplication
        x_t = node_features.transpose(2, 1)
        cov = torch.matmul(node_features, x_t)
        cov_with_zero_diagonal = torch.abs(cov) - torch.abs(torch.diag_embed(torch.diagonal(cov, dim1=-2, dim2=-1)))

        # x_norm = node_features / torch.norm(node_features, dim=-1, keepdim=True)
        # cov = torch.matmul(x_norm, x_norm.transpose(-2, -1)) / x_norm.shape[-1]
        # diag = torch.diag_embed(torch.ones(x_norm.shape[1], device=x.device))
        # cov_with_zero_diagonal = cov * (1 - diag) + diag


        # Initialize final features array and sum of off-diagonal covariance elements
        final_features = torch.zeros((batch_size, k, feat_dim)).to(device=x.device)
        #selected_nodes=[]
        sums = torch.sum(cov_with_zero_diagonal, dim=-1)

        # Iterate to select features and apply Gram-Schmidt orthogonalization
        for i in range(k):
            node = torch.argmin(sums, dim=1) # Find node with lowest sum of off-diagonal covariance elements
            node_features_selected = node_features[torch.arange(node_features.shape[0]), node, :]
            final_features[:, i] = node_features_selected # Add selected features to final feature set
            #sums += cov_with_zero_diagonal.transpose(2,1)[torch.arange(batch_size), node.squeeze(), :]
            sums+= cov_with_zero_diagonal[torch.arange(node_features.shape[0]), node, :]

            # Replace the minimum value with a large value
            sums[torch.arange(node_features.shape[0]), node] = float('inf')

            #selected_nodes.append(node)
        
        final_features = gram_schmidt(final_features)
        
        return final_features.reshape(batch_size, -1)
 

def gram_schmidt(vectors):
    basis = torch.empty(0, device='cuda')
    basis = basis.tolist()
    for v in vectors:
        w = v - sum(torch.dot(v.flatten(), b.flatten())*b for b in basis)
        if torch.norm(w) > 1e-10:
            basis.append(w/torch.norm(w))
    return torch.stack(basis)

class ModuleSERO(nn.Module):
    def __init__(self, output_dim, hidden_dim, dropout=0.1, upscale=1.0):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(hidden_dim, round(upscale*hidden_dim)), nn.BatchNorm1d(round(upscale*hidden_dim)), nn.GELU())
        self.attend = nn.Linear(round(upscale*hidden_dim), output_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, node_axis=1):
        # assumes shape [... x node x ... x feature]
        x = x.reshape(64,-1,x.shape[-1])
        x_readout = x.mean(node_axis)
        x_shape = x_readout.shape
        x_embed = self.embed(x_readout.reshape(-1,x_shape[-1]))
        x_graphattention = torch.sigmoid(self.attend(x_embed)).view(*x_shape[:-1],-1)
        permute_idx = list(range(node_axis))+[len(x_graphattention.shape)-1]+list(range(node_axis,len(x_graphattention.shape)-1))
        x_graphattention = x_graphattention.permute(permute_idx)
        return (x * self.dropout(x_graphattention.unsqueeze(1))).mean(node_axis), x_graphattention
    

class ModuleGARO(nn.Module):
    def __init__(self, output_dim, hidden_dim, dropout=0.1, upscale=1.0):
        super().__init__()
        self.embed_query = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.embed_key = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, node_axis=1):
        # assumes shape [... x node x ... x feature]
        x_q = self.embed_query(x.mean(node_axis, keepdims=True))
        x_k = self.embed_key(x)
        x_graphattention = torch.sigmoid(torch.matmul(x_q, rearrange(x_k, 't b n c -> t b c n'))/np.sqrt(x_q.shape[-1])).squeeze(2)
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1,0,2)
