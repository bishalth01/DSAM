from sys import exit
from typing import Dict, Any
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
from math import ceil
from torch.nn import BatchNorm1d, ModuleList
from torch.nn import Sequential, ReLU, Linear
from torch_geometric.nn import  DenseGraphConv, dense_diff_pool, PNAConv, BatchNorm, DenseSAGEConv, GraphSizeNorm
from torch_geometric.nn import MetaLayer, GraphNorm
from torch_geometric.nn import global_mean_pool, GCNConv, GATConv, global_add_pool, GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean
from utils import ConvStrategy, PoolingStrategy, EncodingStrategy, SweepType
from net.dynamicbraingnn import CustomNetwork
from einops import rearrange, repeat
from torch_sparse import coalesce
from xgboost import XGBClassifier
from torch_geometric.nn import TopKPooling
import xgboost as xgb
from net.MultiSetAttention.mainnet import GraphMultisetTransformer
from torch.multiprocessing import Process, Queue
from tcn import TemporalConvNet

class GNN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 run_cfg,
                 lin=True,
                 aggr='add'):
        super(GNN, self).__init__()

        self.dp_norm = run_cfg['dp_norm']

        if run_cfg['dp_norm'] == 'batchnorm':
            self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
            self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
            self.bn3 = torch.nn.BatchNorm1d(out_channels)
        elif run_cfg['dp_norm'] == 'graphnorm':
            self.bn1 = GraphNorm(hidden_channels)
            self.bn2 = GraphNorm(hidden_channels)
            self.bn3 = GraphNorm(out_channels)
        elif run_cfg['dp_norm'] == 'graphsizenorm':
            self.bn1 = GraphSizeNorm()
            self.bn2 = GraphSizeNorm()
            self.bn3 = GraphSizeNorm()

        self.conv1 = DenseGraphConv(in_channels, hidden_channels, aggr=aggr)
        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels, aggr=aggr)
        self.conv3 = DenseGraphConv(hidden_channels, out_channels, aggr=aggr)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        batch = torch.repeat_interleave(torch.full((batch_size,), num_nodes, dtype=torch.long)).to(x.device)

        x = x.view(-1, num_channels)
        if self.dp_norm == 'batchnorm':
            x = getattr(self, 'bn{}'.format(i))(x)
        else:
            x = getattr(self, 'bn{}'.format(i))(x, batch)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        # Mask will always be true in our case because graphs have all fixed number of nodes.
        x0 = x
        if self.dp_norm == 'nonorm':
            x1 = F.relu(self.conv1(x0, adj, mask))
            x2 = F.relu(self.conv2(x1, adj, mask))
            x3 = F.relu(self.conv3(x2, adj, mask))
        else:
            x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
            x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
            x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x
    
class DiffPoolLayer(torch.nn.Module):
    def __init__(self, max_num_nodes, num_init_feats, aggr, run_cfg):
        super(DiffPoolLayer, self).__init__()
        self.aggr = aggr
        if self.aggr == 'improved':
            aggr = 'add'
        self.init_feats = num_init_feats
        self.max_nodes = max_num_nodes
        self.INTERN_EMBED_SIZE = self.init_feats  # ceil(self.init_feats / 3)

        num_nodes = max(1, ceil(run_cfg['dp_perc_retaining'] * self.max_nodes))
        self.gnn1_pool = GNN(self.init_feats, self.INTERN_EMBED_SIZE, num_nodes, aggr=aggr, run_cfg=run_cfg)
        self.gnn1_embed = GNN(self.init_feats, self.INTERN_EMBED_SIZE, self.INTERN_EMBED_SIZE, lin=False, aggr=aggr, run_cfg=run_cfg)

        num_nodes = max(1, ceil(run_cfg['dp_perc_retaining'] * num_nodes))
        self.final_num_nodes = num_nodes
        self.gnn2_pool = GNN(3 * self.INTERN_EMBED_SIZE, self.INTERN_EMBED_SIZE, num_nodes, aggr=aggr, run_cfg=run_cfg)
        self.gnn2_embed = GNN(3 * self.INTERN_EMBED_SIZE, self.INTERN_EMBED_SIZE, self.INTERN_EMBED_SIZE, lin=False, aggr=aggr, run_cfg=run_cfg)

        self.gnn3_embed = GNN(3 * self.INTERN_EMBED_SIZE, self.INTERN_EMBED_SIZE, self.INTERN_EMBED_SIZE, lin=False, aggr=aggr, run_cfg=run_cfg)
        if self.aggr == 'improved':
            self.final_mlp = nn.Linear(self.final_num_nodes * 3 * self.INTERN_EMBED_SIZE , 3 * self.INTERN_EMBED_SIZE)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)
        if self.aggr == 'add':
            x = x.sum(dim=1)
        elif self.aggr == 'improved':
            x = self.final_mlp(x.reshape(-1, self.final_num_nodes * 3 * self.INTERN_EMBED_SIZE))
        else:
            x = x.mean(dim=1)

        return x, l1 + l2, e1 + e2


class EdgeModel(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, activation='relu'):
        super().__init__()
        self.input_size = 2 * num_node_features + num_edge_features
        dict_activations = {'relu': nn.ReLU(),
                            'elu': nn.ELU(),
                            'tanh': nn.Tanh()}
        self.activation = dict_activations[activation]
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.input_size, int(self.input_size / 2)),
            self.activation,
            nn.Linear(int(self.input_size / 2), num_edge_features),
        )

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], 1)
        out = self.edge_mlp(out)
        return out


class NodeModel(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, activation='relu'):
        super(NodeModel, self).__init__()
        self.input_size = num_node_features + num_edge_features
        dict_activations = {'relu': nn.ReLU(),
                            'elu': nn.ELU(),
                            'tanh': nn.Tanh()}
        self.activation = dict_activations[activation]

        self.node_mlp_1 = nn.Sequential(
            nn.Linear(self.input_size, self.input_size * 2),
            self.activation,
            nn.Linear(self.input_size * 2, self.input_size * 2),
        )
        self.node_mlp_2 = nn.Sequential(
            nn.Linear(num_node_features + self.input_size * 2, self.input_size),
            self.activation,
            nn.Linear(self.input_size, num_node_features),
        )

    def forward(self, x, edge_index, edge_attr):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        # Scatter around "col" (destination nodes)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        # Concatenate X with transformed representation given the source nodes with edge's messages
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)


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
        #self.sero = ModuleSERO(hidden_dim=num_node_features, input_dim=num_node_features, dropout=0.1, upscale=1.0)
        for _ in range(run_cfg['nodemodel_layers']):
            conv = PNAConv(in_channels=num_node_features, out_channels=num_node_features,
                           aggregators=aggregators, scalers=scalers, deg=run_cfg['dataset_indegree'],
                           edge_dim=num_edge_features, towers=1, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(num_node_features))
        

    def forward(self, x, edge_index, edge_attr, u=None, batch=None, pos=None):
        all_outputs_minmax=[]
        #all_outputs_sero=[]
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = self.activation(batch_norm(conv(x, edge_index, edge_attr)))
            #all_outputs_minmax.append(torch.cat([gmp(x, batch), gap(x, batch)], dim=1))
            #new_attention_features, attentiongraph = self.sero(x)
            #all_outputs_sero.append(new_attention_features)

        return x#, torch.cat(all_outputs_minmax, dim=1)


class PNANodeModelWithPool(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, activation, run_cfg, pool_ratio):
        super(PNANodeModelWithPool, self).__init__()

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
        # self.sero = ModuleSERO(hidden_dim=num_node_features, input_dim=num_node_features, dropout=0.1, upscale=1.0)
        self.pools=ModuleList()
        for _ in range(run_cfg['nodemodel_layers']):
            conv = PNAConv(in_channels=num_node_features, out_channels=num_node_features,
                           aggregators=aggregators, scalers=scalers, deg=run_cfg['dataset_indegree'],
                           edge_dim=num_edge_features, towers=1, pre_layers=1, post_layers=1,
                           divide_input=False)
            pool = TopKPooling(num_node_features, ratio=pool_ratio, multiplier=1, nonlinearity=torch.sigmoid)
            self.convs.append(conv)
            self.pools.append(pool)
            self.batch_norms.append(BatchNorm(num_node_features))
        

    def forward(self, x, edge_index, edge_attr, u=None, batch=None, pos=None):
        all_outputs_minmax=[]
        for conv, batch_norm, pools in zip(self.convs, self.batch_norms, self.pools):
            pool_output, edge_index, edge_attr, batch, perm, score1 = pools(conv(x, edge_index, edge_attr), edge_index, edge_attr, batch)
            x = self.activation(batch_norm(pool_output))
            all_outputs_minmax.append(torch.cat([gmp(x, batch), gap(x, batch)], dim=1))
            
        return x, torch.cat(all_outputs_minmax, dim=1)

# Define the Transformer-based attention model
class TransformerAttention(nn.Module):
    def __init__(self, input_size, num_heads):
        super(TransformerAttention, self).__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.attention_layer = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads)

    def forward(self, x):
        # x: (seq_len, batch_size, input_size)
        x = x.permute(2, 0, 1)  # (input_size, seq_len, batch_size)
        attn_output, _ = self.attention_layer(x, x, x)
        attn_output = attn_output.permute(1, 2, 0)  # (seq_len, batch_size, input_size)
        return attn_output
    

def min_max_normalization(x, min_val, max_val):
    normalized_x = (x - min_val) / (max_val - min_val)
    return normalized_x


class SelfAttentionFNC(nn.Module):
    def __init__(self, tcn_output_dim, attention_embedding, num_attention_heads, device):
        super(SelfAttentionFNC, self).__init__()
        self.tcn_output_dim = tcn_output_dim
        self.attention_embedding = attention_embedding
        self.num_attention_heads = num_attention_heads
        self.device = device

        self.key_layer = nn.Sequential(
            nn.Linear(self.tcn_output_dim, self.attention_embedding),
        ).to(self.device)

        self.value_layer = nn.Sequential(
            nn.Linear(self.tcn_output_dim, self.attention_embedding),
        ).to(self.device)

        self.query_layer = nn.Sequential(
            nn.Linear(self.tcn_output_dim, self.attention_embedding),
        ).to(self.device)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.attention_embedding,
            num_heads=self.num_attention_heads
        ).to(self.device)
        

    def forward(self, outputs):
        key = self.key_layer(outputs)
        value = self.value_layer(outputs)
        query = self.query_layer(outputs)

        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)
        query = query.permute(1, 0, 2)

        attn_output, attn_output_weights = self.multihead_attn(key, value, query)

        attn_output = attn_output.permute(1, 0, 2)

        attn_output_weights = attn_output_weights  # + FNC + FNC2
        return attn_output, attn_output_weights



class SpatioTemporalModel(nn.Module):
    def __init__(self, run_cfg: Dict[str, Any],
                 multimodal_size: int = 0, model_version: str = '80',
                 encoding_model=None):
        super(SpatioTemporalModel, self).__init__()

        num_time_length = run_cfg['time_length']
        #num_time_length = 184
        dropout_perc = run_cfg['param_dropout']
        pooling = run_cfg['param_pooling']
        channels_conv = run_cfg['param_channels_conv']
        activation = run_cfg['param_activation']
        conv_strategy = run_cfg['param_conv_strategy']
        sweep_type = run_cfg['sweep_type']
        gat_heads = run_cfg['param_gat_heads']
        edge_weights = run_cfg['edge_weights']
        final_sigmoid = run_cfg['model_with_sigmoid']
        num_nodes = run_cfg['num_nodes']
        num_gnn_layers = run_cfg['param_num_gnn_layers']
        encoding_strategy = run_cfg['param_encoding_strategy']
        multimodal_size = run_cfg['multimodal_size']
        temporal_embed_size = run_cfg['temporal_embed_size']
        threshold = run_cfg['param_threshold']
        batch_size = run_cfg['batch_size']
        attention_threshold=run_cfg['attention_threshold']
        number_hidden_units = run_cfg['tcn_hidden_units']
        #BrainGNN
        lamb0= run_cfg['lamb0'] 
        lamb1= run_cfg['lamb1'] 
        lamb2= run_cfg['lamb2'] 
        lamb3= run_cfg['lamb3'] 
        lamb4= run_cfg['lamb4'] 
        lamb5= run_cfg['lamb5'] 
        layer= run_cfg['layer'] 
        n_bgnn_layers= run_cfg['n_layers']
        n_fc_layers = run_cfg['n_fc_layers']
        n_clustered_communities= run_cfg['n_clustered_communities']
        bgnn_ratio= run_cfg['bgnnratio']
        fc_dropout = run_cfg['fc_dropout']

        self.VERSION = model_version

        #Convert Number of Layers into Int List
        #FOR JOB
        # opt.n_layers = [int(float(numeric_string)) for numeric_string in opt.n_layers.split(',')]
        # opt.n_fc_layers = [int(float(numeric_string)) for numeric_string in opt.n_fc_layers.split(',')]



        #if pooling not in [PoolingStrategy.MEAN, PoolingStrategy.DIFFPOOL, PoolingStrategy.CONCAT]:
        #    print('THIS IS NOT PREPARED FOR OTHER POOLING THAN MEAN/DIFFPOOL/CONCAT')
        #    exit(-1)
        if conv_strategy not in [ConvStrategy.TCN_ENTIRE, ConvStrategy.CNN_ENTIRE, ConvStrategy.NONE, ConvStrategy.LSTM, ConvStrategy.TRANSFORMER, ConvStrategy.STAGIN]:
            print('THIS IS NOT PREPARED FOR THAT CONV STRATEGY')
            exit(-1)
        if activation not in ['relu', 'tanh', 'elu']:
            print('THIS IS NOT PREPARED FOR OTHER ACTIVATION THAN relu/tanh/elu')
            exit(-1)
        if sweep_type == SweepType.GAT:
            print('GAT is not ready for edge_attr')
            exit(-1)
        if conv_strategy != ConvStrategy.NONE and encoding_strategy not in [EncodingStrategy.NONE,
                                                                            EncodingStrategy.STATS]:
            print('Mismatch on conv_strategy/encoding_strategy')
            exit(-1)

        self.multimodal_size: int = multimodal_size
        self.TEMPORAL_EMBED_SIZE: int = temporal_embed_size
        self.NODE_EMBED_SIZE: int = self.TEMPORAL_EMBED_SIZE + self.multimodal_size

        if self.multimodal_size > 0:
            self.multimodal_lin = nn.Linear(self.multimodal_size, self.multimodal_size)
            self.multimodal_batch = BatchNorm1d(self.multimodal_size)

        self.conv_strategy = conv_strategy
        self.encoding_strategy = encoding_strategy
        self.encoder_model = encoding_model
        if encoding_model is not None:
            self.NODE_EMBED_SIZE = self.encoding_model.EMBED_SIZE
        elif self.conv_strategy == ConvStrategy.NONE:
            self.NODE_EMBED_SIZE = num_time_length

        if self.encoding_strategy == EncodingStrategy.STATS:
            self.stats_lin = nn.Linear(self.TEMPORAL_EMBED_SIZE, self.TEMPORAL_EMBED_SIZE)
            self.stats_batch = BatchNorm1d(self.TEMPORAL_EMBED_SIZE)

        self.dropout: float = dropout_perc
        self.pooling = pooling
        dict_activations = {'relu': nn.ReLU(),
                            'elu': nn.ELU(),
                            'tanh': nn.Tanh()}
        self.activation = dict_activations[activation]
        self.activation_str = activation
        self.num_nodes = num_nodes

        self.channels_conv = channels_conv
        self.final_sigmoid = final_sigmoid
        self.sweep_type = sweep_type

        self.num_time_length = num_time_length
        self.final_feature_size = ceil(self.num_time_length / 2 / 8)

        self.edge_weights = edge_weights
        self.num_gnn_layers = num_gnn_layers
        self.gat_heads = gat_heads
        self.multimodal_size=0
        self.threshold = threshold
        self.temporal_embed_size=temporal_embed_size
        self.batch_size = batch_size
        self.attention_threshold= attention_threshold
        self.number_hidden_units = number_hidden_units
        self.fc_dropout = fc_dropout
        

        #BrainGNN
        self.lamb0=lamb0  
        self.lamb1=lamb1  
        self.lamb2=lamb2  
        self.lamb3=lamb3  
        self.lamb4=lamb4  
        self.lamb5=lamb5  
        self.layer=layer  
        self.n_bgnn_layers=n_bgnn_layers
        self.n_fc_layers=n_fc_layers
        self.n_clustered_communities= n_clustered_communities
        self.bgnnratio = bgnn_ratio

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



        if self.sweep_type == SweepType.GCN:
            self.gnn_conv1 = GCNConv(self.NODE_EMBED_SIZE,
                                     self.NODE_EMBED_SIZE)
            if self.num_gnn_layers == 2:
                self.gnn_conv2 = GCNConv(self.NODE_EMBED_SIZE,
                                         self.NODE_EMBED_SIZE)
        elif self.sweep_type == SweepType.GAT:
            self.gnn_conv1 = GATConv(self.NODE_EMBED_SIZE,
                                     self.NODE_EMBED_SIZE,
                                     heads=self.gat_heads,
                                     concat=False,
                                     dropout=dropout_perc)
            if self.num_gnn_layers == 2:
                self.gnn_conv2 = GATConv(self.NODE_EMBED_SIZE,
                                         self.NODE_EMBED_SIZE,
                                         heads=self.gat_heads if self.gat_heads == 1 else int(self.gat_heads / 2),
                                         concat=False,
                                         dropout=dropout_perc)
            

        elif self.sweep_type == SweepType.META_EDGE_NODE:
            self.meta_layer = MetaLayer(edge_model=EdgeModel(num_node_features=self.NODE_EMBED_SIZE,
                                                             num_edge_features=1,
                                                             activation=activation),
                                        node_model=PNANodeModel(num_node_features=self.NODE_EMBED_SIZE, num_edge_features=1,
                                                                activation=self.activation, run_cfg=run_cfg))
        elif self.sweep_type == SweepType.META_NODE:
            # self.meta_layer = MetaLayer(node_model=NodeModel(num_node_features=self.NODE_EMBED_SIZE * 4,
            #                                                 num_edge_features=1,
            #                                                 activation=activation))
            self.meta_layer = PNANodeModel(num_node_features=self.NODE_EMBED_SIZE , num_edge_features=1,
                                            activation=self.activation, run_cfg=run_cfg)
            
        elif self.sweep_type == SweepType.META_NODE_POOL:
            #self.meta_layer = MetaLayer(node_model=NodeModel(num_node_features=self.NODE_EMBED_SIZE,
            #                                                 num_edge_features=1,
            #                                                 activation=activation))
            self.meta_layer = PNANodeModelWithPool(num_node_features=self.NODE_EMBED_SIZE, num_edge_features=1,
                                           activation=self.activation, run_cfg=run_cfg, pool_ratio=run_cfg['pnapoolratio'])
        
            
        elif self.sweep_type == SweepType.BRAIN_GNN:
            #self.meta_layer = MetaLayer(node_model=NodeModel(num_node_features=self.NODE_EMBED_SIZE,
            #                                                 num_edge_features=1,
            #                                                 activation=activation))
            #FOR LOCAL
            # n_layers = [int(float(numeric_string)) for numeric_string in str(self.n_bgnn_layers[0]).split(',')]
            # n_fc_layers = [int(float(numeric_string)) for numeric_string in str(self.n_fc_layers[0]).split(',')]

            #FOR JOB
            
            n_layers = [int(float(numeric_string)) for numeric_string in self.n_bgnn_layers.split(',')]
            n_fc_layers = [int(float(numeric_string)) for numeric_string in self.n_fc_layers.split(',')]
            
            self.meta_layer = CustomNetwork(self.num_nodes,self.bgnnratio,1,n_layers,n_fc_layers,self.n_clustered_communities, self.fc_dropout).to(device) 


        if self.conv_strategy == ConvStrategy.TCN_ENTIRE:
            if run_cfg['tcn_hidden_units'] == 8:
                self.channels_conv=8
                self.size_before_lin_temporal = 3 * self.num_time_length
                # self.size_before_lin_temporal = self.channels_conv * (2 ** (run_cfg['tcn_depth'] - 1)) * self.num_time_length
            else:
                self.size_before_lin_temporal = run_cfg['tcn_hidden_units'] * self.num_time_length

            tcn_layers = []
            for i in range(run_cfg['tcn_depth']):
                if run_cfg['tcn_hidden_units'] == 8:
                    tcn_layers.append(self.channels_conv * (2 ** i) )
                else:
                    tcn_layers.append(run_cfg['tcn_hidden_units'])

            self.temporal_conv1 = TemporalConvNet(1,
                                                 tcn_layers,
                                                 kernel_size=run_cfg['tcn_kernel1'],
                                                 dropout=self.dropout,
                                                 norm_strategy=run_cfg['tcn_norm_strategy'])
            
            
            original_number_timepoints = int(self.size_before_lin_temporal/self.number_hidden_units)
            final_size = int((self.attention_threshold/100) *  original_number_timepoints)  * self.number_hidden_units

            
            self.fnc_attention_module = SelfAttentionFNC(tcn_output_dim=final_size, attention_embedding=run_cfg['fnc_embed_dim'], num_attention_heads=run_cfg['fnc_attnhead'], device=device)
            self.timepointsattention = TransformerAttention(self.num_nodes, run_cfg['timepoints_attnhead']).to(device)

        elif self.conv_strategy == ConvStrategy.LSTM:
            self.temporal_conv = nn.LSTM(input_size=1,
                                         hidden_size=run_cfg['tcn_hidden_units'],
                                         num_layers=run_cfg['tcn_depth'],
                                         dropout=dropout_perc,
                                         batch_first=True)

            self.size_before_lin_temporal = run_cfg['tcn_hidden_units'] * self.num_time_length
            self.lin_temporal = self._get_lin_temporal(run_cfg)

            def init_lstm_hidden(x):
                h0 = torch.zeros(run_cfg['tcn_depth'], x.size(0), run_cfg['tcn_hidden_units'])
                c0 = torch.zeros(run_cfg['tcn_depth'], x.size(0), run_cfg['tcn_hidden_units'])
                return [t.to(x.device) for t in (h0, c0)]

            self.init_lstm_hidden = init_lstm_hidden
        

        if self.pooling == PoolingStrategy.DIFFPOOL:
            self.pre_final_linear = nn.Linear(3 * self.NODE_EMBED_SIZE, self.NODE_EMBED_SIZE)

            self.diff_pool = DiffPoolLayer(num_nodes, self.NODE_EMBED_SIZE, aggr='mean', run_cfg=run_cfg)
        elif self.pooling == PoolingStrategy.CONCAT:
            self.pre_final_linear = nn.Linear(self.num_nodes * self.NODE_EMBED_SIZE, self.NODE_EMBED_SIZE)
            
        elif self.pooling == PoolingStrategy.MAXMIN:
            self.pre_final_linear = nn.Linear(self.NODE_EMBED_SIZE * run_cfg['nodemodel_layers'] * 2, self.NODE_EMBED_SIZE)
            
        elif self.pooling in [PoolingStrategy.DP_MAX, PoolingStrategy.DP_ADD, PoolingStrategy.DP_MEAN, PoolingStrategy.DP_IMPROVED]:
            self.pre_final_linear = nn.Linear(3 * self.NODE_EMBED_SIZE, self.NODE_EMBED_SIZE)
            print(f'Special DiffPool: {self.pooling}.')

            if self.pooling == PoolingStrategy.DP_MAX:
                self.diff_pool = DiffPoolLayer(num_nodes, self.NODE_EMBED_SIZE, aggr='max', run_cfg=run_cfg)
            elif self.pooling == PoolingStrategy.DP_ADD:
                self.diff_pool = DiffPoolLayer(num_nodes, self.NODE_EMBED_SIZE, aggr='add', run_cfg=run_cfg)
            elif self.pooling == PoolingStrategy.DP_MEAN:
                self.diff_pool = DiffPoolLayer(num_nodes, self.NODE_EMBED_SIZE, aggr='mean', run_cfg=run_cfg)
            elif self.pooling == PoolingStrategy.DP_IMPROVED:
                self.diff_pool = DiffPoolLayer(num_nodes, self.NODE_EMBED_SIZE, aggr='improved', run_cfg=run_cfg)


    def init_weights(self):
        self.conv1d_1.weight.data.normal_(0, 0.01)
        self.conv1d_2.weight.data.normal_(0, 0.01)
        self.conv1d_3.weight.data.normal_(0, 0.01)
        self.conv1d_4.weight.data.normal_(0, 0.01)

    


    def run_tcn_gnn_model(self, data, x, edge_index, edge_attr):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Processing temporal part
        if self.conv_strategy != ConvStrategy.NONE:
            if self.conv_strategy == ConvStrategy.LSTM:
                x = x.view(-1, self.num_time_length, 1)
                h0, c0 = self.init_lstm_hidden(x)
                x, (_, _) = self.temporal_conv1(x, (h0, c0))
                x = x.contiguous()
            else:
                x = x.view(-1, 1, self.num_time_length)
                #x = self.temporal_conv(x)
                # Pass the input through the parallel networks
                x = self.temporal_conv1(x)

                
            batch_size = int(data.batch.shape[0]/self.num_nodes)                
            # Reshape the tensor
            x = x.view(batch_size, self.num_nodes, 3, self.num_time_length)

            # Permute dimensions to get the desired shape
            x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, self.num_nodes, -1)
            original_number_timepoints = self.num_time_length

            # #_-----------------------------------Transformer Attention-----------------------------------

            # Apply the Transformer-based attention separately for each set of 1200 time points
            transformer_outputs = []
            # for i in range(self.number_hidden_units):
            for i in range(3):
                start_idx = i * original_number_timepoints
                end_idx = (i + 1) * original_number_timepoints
                transformer_output = self.timepointsattention(x[:, :, start_idx:end_idx])
                transformer_outputs.append(transformer_output.detach())
            

            # List to store the masked outputs for each set of 1200 time points
            masked_outputs = []

            # for i in range(self.number_hidden_units):
            for i in range(3):                
                start_idx = i * original_number_timepoints
                end_idx = (i + 1) * original_number_timepoints
                attention_output = transformer_outputs[i]
                attention_scores = torch.mean(attention_output, dim=1)  # (batch_size, num_timepoints)
                # Find the number of time points to keep (top 70%)
                num_timepoints_to_keep = int((self.attention_threshold/100) * original_number_timepoints)

                # Find the indices of the top 70% time points based on attention scores
                _, top_indices = torch.topk(attention_scores, num_timepoints_to_keep, dim=1)

                # Sort the indices in ascending order
                top_indices, _ = torch.sort(top_indices, dim=1)

                # Extract 70% of values for each set of four 1200 time points
                expanded_top_indices = top_indices.unsqueeze(1).expand(-1, x.shape[1], -1)

                # Use advanced indexing to extract the points from x based on the indices in top_indices
                extracted_points = torch.gather(x, 2, expanded_top_indices)

                masked_outputs.append(extracted_points)
        

            # Concatenate the masked outputs back together along the time dimension (1200)
            masked_output_combined = torch.cat(masked_outputs, dim=2)

            atten_output, corr = self.fnc_attention_module(masked_output_combined)

            x = corr.reshape(-1, corr.shape[-1])

        elif self.encoding_strategy == EncodingStrategy.STATS:
            x = self.stats_lin(x)
            x = self.activation(x)
            x = self.stats_batch(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        elif self.encoding_strategy == EncodingStrategy.VAE3layers:
            mu, logvar = self.encoder_model.encode(x)
            x = self.encoder_model.reparameterize(mu, logvar)
        elif self.encoding_strategy == EncodingStrategy.AE3layers:
            x = self.encoder_model.encode(x)

        if self.multimodal_size > 0:
            x = torch.cat((xn, x), dim=1)

        if self.sweep_type in [SweepType.GAT, SweepType.GCN]:
            if self.edge_weights:
                x = self.gnn_conv1(x, edge_index, edge_weight=edge_attr.view(-1))
            else:
                x = self.gnn_conv1(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, training=self.training)
            if self.num_gnn_layers == 2:
                if self.edge_weights:
                    x = self.gnn_conv2(x, edge_index, edge_weight=edge_attr.view(-1))
                else:
                    x = self.gnn_conv2(x, edge_index)
                x = self.activation(x)
                x = F.dropout(x, training=self.training)
        elif self.sweep_type == SweepType.META_NODE:
            #x, alloutputs = self.meta_layer(x, edge_index, edge_attr, batch=data.batch, pos=data.pos)
            #x = self.meta_layer(x, edge_index, edge_attr, batch=data.batch)
            x = self.meta_layer(x, edge_index, edge_attr, batch=data.batch)
        elif self.sweep_type == SweepType.META_NODE_POOL:
            x, alloutputs = self.meta_layer(x, edge_index, edge_attr, batch=data.batch, pos=data.pos)
        elif self.sweep_type == SweepType.META_EDGE_NODE:
            x, alloutputs, edge_attr, _ = self.meta_layer(x, edge_index, edge_attr, batch = data.batch)
        elif self.sweep_type == SweepType.BRAIN_GNN:
            x, allpools, scores, edge_attr = self.meta_layer(x, edge_index, data.batch, edge_attr, data.pos)
            #x = torch.cat([x, minmaxscores], dim=1)
        elif self.sweep_type == SweepType.GRAPH_MULTISET_TRANSFORMER:
            x, all_concatenated = self.meta_layer(x, edge_index, data.batch)
            #x = torch.cat([x, minmaxscores], dim=1)
        elif self.sweep_type == SweepType.GIN:
            x, all_concatenated = self.meta_layer(x, edge_index, edge_attr, data.batch)
        
        elif self.sweep_type == SweepType.GATED_GRAPH:
            x = self.meta_layer( data, x, edge_index, edge_attr, data.batch)

        
        if self.sweep_type != SweepType.BRAIN_GNN and self.sweep_type != SweepType.GIN and self.sweep_type != SweepType.GRAPH_MULTISET_TRANSFORMER and  self.sweep_type != SweepType.GATED_GRAPH:

            if self.pooling == PoolingStrategy.MEAN:
                x = global_mean_pool(x, data.batch)
            elif self.pooling == PoolingStrategy.ADD:
                x = global_add_pool(x, data.batch)
            elif self.pooling in [PoolingStrategy.DIFFPOOL, PoolingStrategy.DP_MAX, PoolingStrategy.DP_ADD, PoolingStrategy.DP_MEAN, PoolingStrategy.DP_IMPROVED]:
                adj_tmp = pyg_utils.to_dense_adj(edge_index, data.batch, edge_attr=edge_attr)
                if edge_attr is not None: # Because edge_attr only has 1 feature per edge
                    adj_tmp = adj_tmp[:, :, :, 0]
                x_tmp, batch_mask = pyg_utils.to_dense_batch(x, data.batch)

                x, link_loss, ent_loss = self.diff_pool(x_tmp, adj_tmp, batch_mask)

                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.activation(self.pre_final_linear(x))
            elif (self.pooling == PoolingStrategy.CONCAT) and (self.sweep_type != SweepType.META_NODE_POOL):
                x, _ = to_dense_batch(x, data.batch)
                x = x.view(-1, self.NODE_EMBED_SIZE * self.num_nodes)
                x = self.activation(self.pre_final_linear(x))

                x = F.dropout(x, p=self.fc_dropout, training=self.training)
                #x = torch.cat([x, alloutputs], dim=1)

            elif self.pooling == PoolingStrategy.GARO:
                x, graph_attention = self.sero(x, data.batch)
            elif (self.pooling == PoolingStrategy.MAXMIN) or (self.sweep_type == SweepType.META_NODE_POOL):
                #x = torch.cat([x, alloutputs], dim=1)
                x = self.activation(alloutputs)
                #x = F.dropout(x, p=self.dropout, training=self.training)
        
        elif self.sweep_type==SweepType.GRAPH_MULTISET_TRANSFORMER:
            concatenated_features = all_concatenated
            concatenated_features, _ = to_dense_batch(concatenated_features, data.batch)
            concatenated_features = concatenated_features.view(-1, 128 * self.num_nodes )
            concatenated_features = self.activation(self.pre_final_linear(concatenated_features))

            x  = torch.cat([concatenated_features,x], dim=1)
        
        
  
        if self.sweep_type == SweepType.BRAIN_GNN:
            return x, allpools, scores
        
        return x


    def forward(self, data, v=None, a=None, t=None, sampling_endpoints=None):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        x, edge_index, edge_attr = x.to(device), edge_index.to(device), edge_attr.to(device)

        output_1, allpools_1, scores_1 = self.run_tcn_gnn_model(data, x, edge_index, edge_attr)

        return output_1, allpools_1, scores_1
    

    def to_string_name(self):
        model_vars = ['V_' + self.VERSION,
                      'TL_' + str(self.num_time_length),
                      'D_' + str(self.dropout),
                      'A_' + self.activation_str,
                      'P_' + self.pooling.value[:3],
                      'CS_' + self.conv_strategy.value[:3],
                      'CH_' + str(self.channels_conv),
                      'FS_' + str(self.final_sigmoid)[:1],
                      'T_' + self.sweep_type.value[:3],
                      'W_' + str(self.edge_weights)[:1],
                      'GH_' + str(self.gat_heads),
                      'GL_' + str(self.num_gnn_layers),
                      'E_' + self.encoding_strategy.value[:3],
                      'M_' + str(self.multimodal_size),
                      'S_' + str(self.TEMPORAL_EMBED_SIZE)
                      ]

        return ''.join(model_vars)


    def _collate_adjacency(self, a, sparsity, sparse=True):
            i_list = []
            v_list = []
            for sample, _dyn_a in enumerate(a):
                for timepoint, _a in enumerate(_dyn_a):
                    thresholded_a = (_a > self.percentile(_a, 100-sparsity))
                    _i = thresholded_a.nonzero(as_tuple=False)
                    _v = torch.ones(len(_i))
                    _i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                    i_list.append(_i)
                    v_list.append(_v)
            _i = torch.cat(i_list).T.to(a.device)
            _v = torch.cat(v_list).to(a.device)
            return torch.sparse.FloatTensor(_i, _v, (a.shape[0]*a.shape[1]*a.shape[2], a.shape[0]*a.shape[1]*a.shape[3]))
            
    