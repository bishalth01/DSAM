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
from nilearn.connectome import ConnectivityMeasure
from torch_geometric.nn import MetaLayer, GraphNorm
from torch_geometric.nn import DenseGraphConv, dense_diff_pool, PNAConv, BatchNorm, DenseSAGEConv, GraphSizeNorm
import networkx as nx
import copy
from torch.nn import BatchNorm1d, ModuleList
from torch_scatter import scatter_mean


class PNANodeModel(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, activation, run_cfg):
        super(PNANodeModel, self).__init__()

        if run_cfg['nodemodel_aggr'] == 'all':
            aggregators = ['mean', 'min', 'max', 'std', 'sum']
        else:
            aggregators = [run_cfg['nodemodel_aggr']]

        if run_cfg['nodemodel_scalers'] == 'all':
            scalers = ['identity', 'amplification', 'attenuation']
        else:
            scalers = ['identity']

        print(f'--> PNANodeModel going with aggregators={aggregators}, scalers={scalers}')

        self.activation = activation
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(run_cfg['nodemodel_layers']):
            conv = PNAConv(in_channels=num_node_features, out_channels=num_node_features,
                           aggregators=aggregators, scalers=scalers, deg=run_cfg['dataset_indegree'],
                           edge_dim=num_edge_features, towers=1, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(num_node_features))

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = self.activation(batch_norm(conv(x, edge_index, edge_attr)))

        return x

class STGNN(nn.Module):
    def __init__(self, cfg):
        super(STGNN, self).__init__()

        self.cfg = cfg

        num_time_length = cfg.dataset.time_points
        dropout_perc = cfg.model.dropout
        pooling = cfg.model.pooling
        activation = cfg.model.activation
        conv_strategy = cfg.model.param_conv_strategy
        sweep_type = cfg.model.sweep_type
        # gat_heads = cfg.model.param_gat_heads
        edge_weights = True
        final_sigmoid = True
        num_nodes = cfg.model.num_nodes
        temporal_embed_size = cfg.model.temporal_embed_size
        threshold = cfg.model.threshold
        batch_size = cfg.dataset.batch_size
        attention_threshold = cfg.model.attentionthreshold
        number_hidden_units = cfg.model.tcn_hidden_units

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

        self.TEMPORAL_EMBED_SIZE: int = temporal_embed_size
        self.NODE_EMBED_SIZE: int = self.TEMPORAL_EMBED_SIZE

        self.conv_strategy = conv_strategy

        self.dropout: float = dropout_perc
        self.pooling = pooling
        dict_activations = {'relu': nn.ReLU(),
                            'elu': nn.ELU(),
                            'tanh': nn.Tanh()}
        self.activation = dict_activations[activation]
        self.activation_str = activation
        self.num_nodes = num_nodes

        # self.channels_conv = channels_conv
        self.final_sigmoid = final_sigmoid
        self.sweep_type = sweep_type

        self.num_time_length = num_time_length
        self.final_feature_size = ceil(self.num_time_length / 2 / 8)

        self.edge_weights = edge_weights
        self.dynamic = cfg.model.dynamic
        # self.num_gnn_layers = num_gnn_layers
        # self.gat_heads = gat_heads
        self.multimodal_size=0
        self.threshold = threshold
        self.temporal_embed_size=temporal_embed_size
        self.batch_size = batch_size
        self.attention_threshold= attention_threshold
        self.number_hidden_units = number_hidden_units
        self.fc_dropout = fc_dropout
        
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

            
        if self.sweep_type == SweepType.BRAIN_GNN:

            # #FOR LOCAL
            # n_layers = [int(float(numeric_string)) for numeric_string in str(self.n_bgnn_layers[0]).split(',')]
            # n_fc_layers = [int(float(numeric_string)) for numeric_string in str(self.n_fc_layers[0]).split(',')]

            #FOR JOB
            n_layers = [int(float(numeric_string)) for numeric_string in str(self.n_bgnn_layers[0]).split(',')]
            n_fc_layers = [int(float(numeric_string)) for numeric_string in str(self.n_fc_layers[0]).split(',')]
            
            # self.meta_layer = CustomNetwork(self.num_nodes,self.bgnn_ratio,1,n_layers,n_fc_layers,self.n_clustered_communities, self.fc_dropout).to(device)
            self.meta_layer = PNANodeModel(num_node_features=32, num_edge_features=1,
                                           activation=self.activation, run_cfg=cfg)

        if self.conv_strategy == ConvStrategy.TCN_ENTIRE:
            if cfg.model.tcn_hidden_units == 8:
                self.channels_conv=8
                self.size_before_lin_temporal = 3 * self.num_time_length#self.channels_conv * (2 ** (cfg.model.tcn_depth - 1)) * self.num_time_length
            else:
                self.size_before_lin_temporal = cfg.model.tcn_hidden_units * self.num_time_length

            tcn_layers = []
            for i in range(cfg.model.tcn_depth):
                if cfg.model.tcn_hidden_units == 8:
                    tcn_layers.append(self.channels_conv * (2 ** i) )
                else:
                    tcn_layers.append(cfg.model.tcn_hidden_units)

            self.temporal_conv1 = TemporalConvNet(1,
                                                 tcn_layers,
                                                 kernel_size=cfg.model.tcn_kernel1,
                                                 dropout=self.dropout,
                                                 norm_strategy=cfg.model.tcn_norm_strategy)
            original_number_timepoints = int(self.size_before_lin_temporal/self.number_hidden_units)
            # final_size = int((self.attention_threshold/100) *  original_number_timepoints)  * self.number_hidden_units

            final_size = int((self.attention_threshold/100) *  self.num_time_length) * 3

        
        
        elif self.pooling == PoolingStrategy.CONCAT:
            self.pre_final_linear = nn.Linear(self.num_nodes * self.NODE_EMBED_SIZE, self.NODE_EMBED_SIZE)

    def run_tcn_gnn_model(self, data, x, edge_index, edge_attr):
            # Ensure all operations are performed on GPU
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

            #Performing multilevel temporal feature capture
            x = x.view(-1, 1, self.num_time_length).to(device)
            x = self.temporal_conv1(x)

            x = x.view(x.size()[0], -1)
            x = self.lin_temporal(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            
           
            # Apply Meta-layer
            x = corr.reshape(-1, corr.shape[-1]).to(device)
            # x, allpools, scores, edge_attr = self.meta_layer(x, edge_index, data.batch, new_edge_attr, data.pos)
            x, allpools, scores, edge_attr = self.meta_layer(x, edge_index, edge_attr)
            

            return x, allpools, scores

    def forward(self, data, v=None, a=None, t=None, sampling_endpoints=None):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        data = data.to(device)
        x, edge_index, edge_attr = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)

        output_1, allpools_1, scores_1 = self.run_tcn_gnn_model(data, x, edge_index, edge_attr)
        return output_1, allpools_1, scores_1
