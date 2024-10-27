# import torch
# import torch.nn.functional as F
# from torch.nn import Parameter
# from dsamcomponents.models.brainmsgpassing import MyMessagePassing
# from torch_geometric.utils import add_remaining_self_loops,softmax

# from torch_geometric.typing import (OptTensor)

# from dsamcomponents.models.inits import uniform
# import numpy as np


# class MyGINConv(MyMessagePassing):
#     def __init__(self, in_channels, out_channels, nn, normalize=False, bias=True, eps=0,
#                  **kwargs):
#         super(MyGINConv, self).__init__(aggr='add', **kwargs)

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.normalize = normalize
#         self.nn = nn
#         #self.eps = eps
#         self.eps = torch.nn.Parameter(torch.Tensor([eps]))
#         #self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
#         self.mlp = torch.nn.Sequential(torch.nn.Linear(self.out_channels, self.out_channels),
#                                         torch.nn.BatchNorm1d(self.out_channels),
#                                         torch.nn.ReLU(),
#                                         torch.nn.Linear(self.out_channels, self.out_channels))

#         if bias:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()

#     def reset_parameters(self):
# #        uniform(self.in_channels, self.weight)
#         uniform(self.in_channels, self.bias)
#         self.eps.data.fill_(0.0)

#     def forward(self, x, edge_index, edge_weight=None, pseudo= None, size=None):
#         """"""
#         edge_weight = edge_weight.squeeze()
#         if size is None and torch.is_tensor(x):
#             edge_index, edge_weight = add_remaining_self_loops(
#                 edge_index, edge_weight, 1, x.size(0))

#         weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
#         if torch.is_tensor(x):
#             x = torch.matmul(x.unsqueeze(1), weight).squeeze(1)  #W_i*h_i
#         else:
#             x = (None if x[0] is None else torch.matmul(x[0].unsqueeze(1), weight).squeeze(1),
#                  None if x[1] is None else torch.matmul(x[1].unsqueeze(1), weight).squeeze(1))
            
#         out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, size=size, x=x, edge_weight=edge_weight))
#         #out = self.propagate(edge_index, size=size, x=x, edge_weight=edge_weight)

#         return out

#     def message(self, edge_index_i, size_i, x_j, edge_weight, ptr: OptTensor):
#         edge_weight = softmax(edge_weight, edge_index_i, ptr, size_i)
#         return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

#     # def update(self, aggr_out):
#     #     if self.bias is not None:
#     #         aggr_out = aggr_out + self.bias
#     #     if self.normalize:
#     #         aggr_out = F.normalize(aggr_out, p=2, dim=-1)
#     #     return aggr_out

#     def update(self, aggr_out):
#         # learnable weights 
#         if self.bias is not None:
#             aggr_out = aggr_out + self.bias
            
#         #aggr_out = self.nn((1 + self.eps) * aggr_out)
#         if self.normalize:
#             aggr_out = F.normalize(aggr_out, p=2, dim=-1)

#         return aggr_out
    
    
#     # def update(self, aggr_out, x):
#     #     learnable_weights = self.bn(self.lin(x))
#     #     out = aggr_out + learnable_weights
#     #     out = torch.nn.functional.relu(out)
#     #     return out
    
#     def __repr__(self):
#         return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
#                                    self.out_channels)


import torch
import torch.nn.functional as F
from torch.nn import Parameter
from dsamcomponents.models.brainmsgpassing import MyMessagePassing
from torch_geometric.utils import add_remaining_self_loops,softmax

from torch_geometric.typing import (OptTensor)

from dsamcomponents.models.inits import uniform
import numpy as np


class MyGINConv(MyMessagePassing):
    def __init__(self, in_channels, out_channels, nn, normalize=False, bias=True, eps=0,
                 **kwargs):
        super(MyGINConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.epsilon = Parameter(torch.Tensor(1).fill_(eps))  # Learnable epsilon

        # Existing nn for initial weight transformation
        self.nn = nn
        
        mlp_update = torch.nn.Sequential(torch.nn.Linear(self.out_channels, self.out_channels * 2),
                                        torch.nn.BatchNorm1d(self.out_channels * 2),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(self.out_channels * 2, self.out_channels))
        self.mlp_update = mlp_update

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            self.bias.data.zero_()
        self.epsilon.data.fill_(0.0)  # Reset epsilon to initial value

    def forward(self, x, edge_index, edge_weight=None, pseudo=None, size=None):
        if size is None and torch.is_tensor(x):
            edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 1, x.size(0))

        # Apply the nn to transform input features
        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)

        # Transform node features with the weights
        if torch.is_tensor(x):
            x_transformed = torch.matmul(x.unsqueeze(1), weight).squeeze(1)  # W_i * h_i
        else:
            x_transformed = (
                None if x[0] is None else torch.matmul(x[0].unsqueeze(1), weight).squeeze(1),
                None if x[1] is None else torch.matmul(x[1].unsqueeze(1), weight).squeeze(1)
            )
        
        # Apply message passing
        aggr_out = self.propagate(edge_index, size=size, x=x_transformed, edge_weight=edge_weight)

        # Combine with transformed node features and apply MLP
        updated_features = (1 + self.epsilon) * x_transformed + aggr_out
        out = self.mlp_update(updated_features)

        if self.bias is not None:
            out = out + self.bias

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        
        return out


    def message(self, edge_index_i, size_i, x_j, edge_weight, ptr: OptTensor):
        # Apply edge weight normalization if provided
        if edge_weight is not None:
            edge_weight = softmax(edge_weight, edge_index_i, ptr, size_i)
            return edge_weight.view(-1, 1) * x_j
        else:
            return x_j

    def update(self, aggr_out):
        # Apply the learnable bias
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        
        # MLP is applied in the forward function
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, epsilon={:.4f})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.epsilon.item()
        )


