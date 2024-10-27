from sys import exit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from torch_geometric.nn import global_mean_pool, GCNConv, GATConv, global_add_pool, GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import to_dense_batch
from dsamcomponents.tcn import TemporalConvNet
from dsamcomponents.utils import ConvStrategy, PoolingStrategy, EncodingStrategy, SweepType
from dsamcomponents.models.dynamicbraingnn import CustomNetwork
from dsamcomponents.models.braingnnoriginal import Network
import copy

class BrainGNN(nn.Module):
    def __init__(self, cfg):
        super(BrainGNN, self).__init__()

        self.cfg = cfg

        dropout_perc = cfg.model.dropout
        sweep_type = cfg.model.sweep_type

        num_nodes = cfg.model.num_nodes
        batch_size = cfg.dataset.batch_size

        # BrainGNN
        self.lamb0 = cfg.model.lamb0
        self.lamb1 = cfg.model.lamb1
        self.lamb2 = cfg.model.lamb2
        self.lamb3 = cfg.model.lamb3
        self.lamb4 = cfg.model.lamb4
        self.lamb5 = cfg.model.lamb5
        self.layer = cfg.model.layer
        self.n_bgnn_layers = cfg.model.n_layers
        self.n_fc_layers = cfg.model.n_fc_layers
        self.n_clustered_communities = cfg.model.n_clustered_communities
        self.bgnn_ratio = cfg.model.bgnnratio
        fc_dropout = cfg.model.fc_dropout

        #Convert Number of Layers into Int List
        #FOR JOB
        # opt.n_layers = [int(float(numeric_string)) for numeric_string in opt.n_layers.split(',')]
        # opt.n_fc_layers = [int(float(numeric_string)) for numeric_string in opt.n_fc_layers.split(',')]


        self.dropout: float = dropout_perc
        self.num_nodes = num_nodes

        # self.channels_conv = channels_conv
        self.sweep_type = sweep_type

        self.batch_size = batch_size
        self.fc_dropout = fc_dropout
        
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

            
        if self.sweep_type == SweepType.BRAIN_GNN:

            # #FOR LOCAL
            # n_layers = [int(float(numeric_string)) for numeric_string in str(self.n_bgnn_layers[0]).split(',')]
            # n_fc_layers = [int(float(numeric_string)) for numeric_string in str(self.n_fc_layers[0]).split(',')]

            #FOR JOB
            n_layers = [int(float(numeric_string)) for numeric_string in self.n_bgnn_layers.split(',')]
            n_fc_layers = [int(float(numeric_string)) for numeric_string in self.n_fc_layers.split(',')]
            
            self.meta_layer = Network(self.num_nodes,self.bgnn_ratio,1,n_layers,n_fc_layers,self.n_clustered_communities, self.fc_dropout).to(device)


    def forward(self, data):
        # Unpack data
        time_series, edge_index, edge_attr, node_feature, pseudo_torch, batch = (
            data.x, data.edge_index, data.edge_attr, 
            torch.tensor(np.array(data.pearson_corr)), data.pos, data.batch
        )

        # Ensure time series reshaping if needed
        batch_size = time_series.shape[0]

        # Copy tensors and convert to CUDA if available, with appropriate precision
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float32 if device == 'cuda:0' and torch.cuda.get_device_capability(device)[0] >= 7 else torch.float32

        # Convert to tensors and move to the correct device, ensuring appropriate data types
        x_tensor = torch.tensor(node_feature, device=device, dtype=dtype).reshape(-1, node_feature.shape[-1])
        edge_index_tensor = torch.tensor(edge_index, device=device, dtype=torch.long)  # edge_index typically needs to be long/int
        edge_attr_tensor = torch.abs(torch.tensor(edge_attr, device=device, dtype=dtype))
        batch_tensor = torch.tensor(batch, device=device, dtype=torch.long)

        # # Pass tensors through the model
        # output_1, edge_attr, allpools, scores = self.run_tcn_gnn_model(
        #     x_tensor, batch_tensor, edge_index_tensor, edge_attr_tensor, pseudo_torch.to(device, dtype=dtype)
        # )
        x, allpools, scores, edge_attr = self.meta_layer(x_tensor, edge_index_tensor, batch_tensor, edge_attr_tensor, pseudo_torch)

        return x, allpools, scores


    
