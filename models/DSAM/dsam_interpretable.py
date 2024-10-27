from sys import exit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from dsamcomponents.tcn import TemporalConvNet
from torch.nn import TransformerEncoderLayer
from dsamcomponents.utils import ConvStrategy, PoolingStrategy, EncodingStrategy, SweepType
from dsamcomponents.models.dynamicbraingnn import CustomNetwork
from dsamcomponents.models.braingnnoriginal import Network
from nilearn.connectome import ConnectivityMeasure
import networkx as nx
import math
from torch.autograd import Function

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encodings added: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x
    


# Define the Transformer Model with only Multi-Head Attention (MHA)
class TimePointsTransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout=0.0, max_len=5000):
        super(TimePointsTransformerModel, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x.transpose(1, 2)  # Change shape to (batch_size, sequence_length, d_model)
        attn_output, _ = self.multihead_attn(x, x, x)  # Self-attention
    
        # Apply layer normalization (optional)
        output = self.layer_norm(attn_output + x)
        output = output.transpose(1, 2)
        return output


class FNCCustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, q_dim):
        super(FNCCustomMultiheadAttention, self).__init__()
        
        # Ensure the embedding dimension is divisible by the number of heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."
        k_dim = q_dim
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Dimension per head
        
        # Linear layers to project query, key, and value
        self.query_proj = nn.Linear(self.head_dim, q_dim)
        self.key_proj = nn.Linear(self.head_dim, k_dim)
        
    def forward(self, x):
        batch_size, seq_size, embed_dim = x.size()  # Expected input: (batch_size, seq_len, embed_dim)
        
        # Reshape input to separate into heads
        x = x.view(batch_size, seq_size, self.num_heads, self.head_dim)  # Shape: (batch_size, seq_len, num_heads, head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        
        # Project query, key, and value
        q = self.query_proj(x)  # Shape: (batch_size, seq_len, num_heads, q_dim)
        k = self.key_proj(x)    # Shape: (batch_size, seq_len, num_heads, k_dim)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)  # Shape: (batch_size, num_heads, seq_len, seq_len)
        attn_weights = F.softmax(attn_weights, dim=-1)  # Apply softmax to attention weights

        attn_weights = attn_weights.mean(dim=1)


        return attn_weights

    
class SpatioTemporalModel(nn.Module):
    def __init__(self, cfg, dataloader):
        super(SpatioTemporalModel, self).__init__()

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
        # fc_dropout = cfg.model.fc_dropout

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
        # self.fc_dropout = fc_dropout

        self.dataloaders = dataloader
        
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

            
        if self.sweep_type == SweepType.BRAIN_GNN:

            # #FOR LOCAL
            # n_layers = [int(float(numeric_string)) for numeric_string in str(self.n_bgnn_layers[0]).split(',')]
            # n_fc_layers = [int(float(numeric_string)) for numeric_string in str(self.n_fc_layers[0]).split(',')]

            #FOR JOB
            n_layers = [int(float(numeric_string)) for numeric_string in str(self.n_bgnn_layers[0]).split(',')]
            n_fc_layers = [int(float(numeric_string)) for numeric_string in str(self.n_fc_layers[0]).split(',')]
            
            self.meta_layer = Network(self.num_nodes,self.bgnn_ratio,1,n_layers,n_fc_layers,self.n_clustered_communities, self.dropout).to(device)
            # self.meta_layer = Network(cfg.model.fnc_embed_dim,self.bgnn_ratio,1,n_layers,n_fc_layers,self.n_clustered_communities, self.dropout).to(device)

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

            size_per_block = int((self.attention_threshold/100) *  self.num_time_length) * 3


            self.timepointattention =  TimePointsTransformerModel(d_model=self.num_nodes, nhead=cfg.model.timepoints_attnhead, num_layers=1)

            self.fnc_attention_module = FNCCustomMultiheadAttention(embed_dim= size_per_block, num_heads=cfg.model.fnc_attnhead, q_dim=cfg.model.fnc_embed_dim)
        
        elif self.pooling == PoolingStrategy.CONCAT:
            self.pre_final_linear = nn.Linear(self.num_nodes * self.NODE_EMBED_SIZE, self.NODE_EMBED_SIZE)

    
    def compute_sfnc_attention(self, x):
        # x: (batch_size, num_nodes, num_timepoints)
        batch_size, num_nodes, num_timepoints = x.shape

        # Reshape to (batch_size, num_timepoints, num_nodes)
        x_transposed = x.permute(0, 2, 1)

        # Compute covariance or correlation across timepoints
        sfnc_matrix = torch.matmul(x_transposed.transpose(1, 2), x_transposed) / (num_timepoints - 1)  # Shape: (batch_size, num_nodes, num_nodes)

        # Normalize to get correlation (optional)
        std_dev = x_transposed.std(dim=1, keepdim=True) + 1e-6
        x_normalized = x_transposed / std_dev
        sfnc_matrix = torch.matmul(x_normalized.transpose(1, 2), x_normalized) / (num_timepoints - 1)

        # Set diagonal elements to 1
        for i in range(batch_size):
            sfnc_matrix[i].fill_diagonal_(1.0)

        return sfnc_matrix
    

    def select_top_k_timepoints(self, main_features, attention_scores, top_k, alpha=0.1):
        """
        Selects the top `top_k` most important timepoints using hard top-k selection with a skip connection.

        Parameters:
            main_features (torch.Tensor): Shape (batch_size, num_nodes, num_timepoints)
            attention_scores (torch.Tensor): Shape (batch_size, num_nodes, num_timepoints)
            top_k (int): Number of top timepoints to select
            alpha (float): Weight for the residual connection

        Returns:
            torch.Tensor: Selected top_k timepoints, Shape: (batch_size, num_nodes, top_k)
        """
        # Calculate importance scores across nodes
        importance_scores = torch.sum(torch.abs(attention_scores), dim=1)/100  # Shape: (batch_size, num_timepoints)

        # Perform hard top-k selection
        topk_scores, topk_indices = torch.topk(importance_scores, top_k, dim=1, largest=True, sorted=False)  # Shape: (batch_size, top_k)

        # Expand indices to gather across num_nodes
        topk_indices_expanded = topk_indices.unsqueeze(1).expand(-1, main_features.size(1), -1)  # Shape: (batch_size, num_nodes, top_k)

        topk_scores_expanded = topk_scores.unsqueeze(1).expand(-1, main_features.size(1), -1)  # Shape: (batch_size, num_nodes, top_k)

        # Gather the top_k features
        combined_features = torch.gather(main_features, dim=2, index=topk_indices_expanded)  # Shape: (batch_size, num_nodes, top_k)

        return combined_features #+ 0.0001 * topk_scores_expanded

    
    def forward(self, data, time_series, edge_index_tensor, edge_attr_tensor, node_feature, pseudo_torch, batch_tensor):
        # # Copy tensors and convert to CUDA if available, with appropriate precision
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        #Performing multilevel temporal feature capture
        x = time_series.view(-1, 1, self.num_time_length)
        x = self.temporal_conv1(x)

        batch_size = len(data.y)#int(data.batch.shape[0] / self.num_nodes)

        # Reshape the tensor
        x = x.view(batch_size, self.num_nodes, 3, self.num_time_length)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, self.num_nodes, -1)
        original_number_timepoints = self.num_time_length


        # # Separate the input into three segments
        low_level = x[:, :, :original_number_timepoints]
        medium_level = x[:, :, original_number_timepoints:original_number_timepoints*2]
        high_level = x[:, :, original_number_timepoints*2:]

        # print(low_level.requires_grad)  # Should print True


        low_transformer_output = self.timepointattention(low_level) 
        medium_transformer_output = self.timepointattention(medium_level)  
        high_transformer_output = self.timepointattention(high_level)  

        # Number of top timepoints to select
        num_timepoints = low_level.size(2)
        top_k = int(num_timepoints * (self.attention_threshold / 100))  

        # Select top 10% timepoints for each feature level
        low_level_topk = self.select_top_k_timepoints(low_level, low_transformer_output, top_k)
        medium_level_topk = self.select_top_k_timepoints(medium_level, medium_transformer_output, top_k)
        high_level_topk = self.select_top_k_timepoints(high_level, high_transformer_output, top_k)

        # # Concatenate the selected timepoints across low, medium, and high levels
        final_output = torch.cat((low_level_topk, medium_level_topk, high_level_topk), dim=2)  # Shape: (batch_size, num_nodes, 3 * top_k)

        # final_output = x

        # time_series = time_series.reshape(batch_size, -1, time_series.shape[-1])
        #Compute sFNC matrix
        sfnc_matrix = self.compute_sfnc_attention(final_output)

        # Attention and correlation analysis
        corr = self.fnc_attention_module(final_output)

        # Update edge attributes based on sfnc_matrix
        if self.dynamic:
            device = sfnc_matrix.device  # Ensure tensors are on the same device

            # Extract source and target nodes for each edge
            edge_src = edge_index_tensor[0, :]  # Shape: (num_edges,)
            edge_dst = edge_index_tensor[1, :]  # Shape: (num_edges,)

            # Get batch indices for each node
            batch_indices_src = batch_tensor[edge_src]  # Shape: (num_edges,)
            batch_indices_dst = batch_tensor[edge_dst]  # Shape: (num_edges,)

            # Ensure that edges connect nodes within the same sample
            if not torch.equal(batch_indices_src, batch_indices_dst):
                raise ValueError("Edges connect nodes from different batches.")

            batch_indices = batch_indices_src  # Since src and dst have the same batch indices

            # Number of nodes per sample
            num_nodes_per_sample = sfnc_matrix.size(1)

            # Convert global node indices to sample-level indices
            node_i_sample = edge_src - batch_indices * num_nodes_per_sample  # Shape: (num_edges,)
            node_j_sample = edge_dst - batch_indices * num_nodes_per_sample  # Shape: (num_edges,)

            # Get the sfnc values for the edges
            sfnc_values = sfnc_matrix[batch_indices, node_i_sample, node_j_sample]  # Shape: (num_edges,)

            # Update edge_attr_tensor
            new_edge_attr = sfnc_values.unsqueeze(1)  # Shape: (num_edges, 1)

        else:
            new_edge_attr = torch.abs(edge_attr_tensor)  

        # Apply Meta-layer
        x = corr.reshape(-1, corr.shape[-1]).to(device)
        # x = atten_output.reshape(-1, atten_output.shape[-1]).to(device)
        
        x, allpools, scores, edge_attr = self.meta_layer(x, edge_index_tensor, batch_tensor, new_edge_attr,pseudo_torch)

        return x, allpools, scores, sfnc_matrix, corr
