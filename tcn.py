# #
# # From https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
# #
# #
# import torch
# import torch.nn as nn
# from torch.nn import BatchNorm1d
# from torch.nn.utils import weight_norm


# class Chomp1d(nn.Module):
#     def __init__(self, chomp_size):
#         super(Chomp1d, self).__init__()
#         self.chomp_size = chomp_size

#     def forward(self, x):
#         return x[:, :, :-self.chomp_size].contiguous()


# class TemporalBlock(nn.Module):
#     def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, norm_strategy, dropout=0.2):
#         super(TemporalBlock, self).__init__()
#         self.chomp1 = Chomp1d(padding)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(dropout)


#         self.chomp2 = Chomp1d(padding)
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(dropout)

#         if norm_strategy ==  'weight':
#             self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
#                                                stride=stride, padding=padding, dilation=dilation))
#             self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
#                                                stride=stride, padding=padding, dilation=dilation))
#             self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
#                                      self.conv2, self.chomp2, self.relu2, self.dropout2)
#         elif norm_strategy == 'batchnorm':
#             self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
#                                                stride=stride, padding=padding, dilation=dilation)
#             self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
#                                                stride=stride, padding=padding, dilation=dilation)
#             self.batch1 = BatchNorm1d(n_outputs)
#             self.batch2 = BatchNorm1d(n_outputs)
#             self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.batch1, self.dropout1,
#                                      self.conv2, self.chomp2, self.relu2, self.batch2, self.dropout2)


#         self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
#         self.relu = nn.ReLU()
#         self.init_weights()

#     def init_weights(self):
#         self.conv1.weight.data.normal_(0, 0.01)
#         self.conv2.weight.data.normal_(0, 0.01)
#         if self.downsample is not None:
#             self.downsample.weight.data.normal_(0, 0.01)

#     def forward(self, x):
#         out = self.net(x)
#         res = x if self.downsample is None else self.downsample(x)
#         return self.relu(out + res)


# class TemporalConvNet(nn.Module):
#     def __init__(self, num_inputs, num_channels, norm_strategy, kernel_size=2, dropout=0.2):
#         super(TemporalConvNet, self).__init__()
#         layers = []
#         num_levels = len(num_channels)
#         for i in range(num_levels):
#             dilation_size = 2 ** i
#             in_channels = num_inputs if i == 0 else num_channels[i-1]
#             out_channels = num_channels[i]
#             layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
#                                      padding=(kernel_size-1) * dilation_size, dropout=dropout, norm_strategy=norm_strategy)]

#         self.network = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.network(x)

#
# From https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
#
#
import torch
import torch.nn as nn
from torch.nn import BatchNorm1d
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, norm_strategy, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)


        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        if norm_strategy ==  'weight':
            self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation))
            self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation))
            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                     self.conv2, self.chomp2, self.relu2, self.dropout2)
        elif norm_strategy == 'batchnorm':
            self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation)
            self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation)
            self.batch1 = BatchNorm1d(n_outputs)
            self.batch2 = BatchNorm1d(n_outputs)
            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.batch1, self.dropout1,
                                     self.conv2, self.chomp2, self.relu2, self.batch2, self.dropout2)


        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# class TemporalConvNet(nn.Module):
#     def __init__(self, num_inputs, num_channels, norm_strategy, kernel_size=2, dropout=0.2):
#         super(TemporalConvNet, self).__init__()
#         layers = []
#         num_levels = len(num_channels) 
#         for i in range(num_levels):
#             dilation_size = 2 ** i #Dilation size increases with each levels
#             in_channels = num_inputs if i == 0 else num_channels[i-1]
#             out_channels = num_channels[i]
#             layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
#                                      padding=(kernel_size-1) * dilation_size, dropout=dropout, norm_strategy=norm_strategy)]

#         self.network = nn.Sequential(*layers)

#         self.low_level_conv = nn.Conv1d(in_channels=num_channels[0], out_channels=1, kernel_size=1)
#         self.medium_level_conv = nn.Conv1d(in_channels=num_channels[1] - num_channels[0], out_channels=1, kernel_size=1)
#         self.high_level_conv = nn.Conv1d(in_channels=num_channels[2] - num_channels[1], out_channels=1, kernel_size=1)


#     def forward(self, x):
#         output = self.network(x)

#         # Split the output into low, medium, and high-level features
#         low_level_features = output[:, :8, :]
#         medium_level_features = output[:, 8:16, :]
#         high_level_features = output[:, 16:, :]

#         # Apply the 1x1 convolutional layers to reduce the channel dimension for each level
#         low_level_reduced = self.low_level_conv(low_level_features)
#         medium_level_reduced = self.medium_level_conv(medium_level_features)
#         high_level_reduced = self.high_level_conv(high_level_features)

#         # Concatenate the reduced features along the channel dimension
#         combined_features = torch.cat([low_level_reduced, medium_level_reduced, high_level_reduced], dim=1)

#         return combined_features
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, norm_strategy, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.num_levels = len(num_channels) 
        self.temporal_blocks = nn.ModuleList()


        # for i in range(num_levels):
        #     dilation_size = 2 ** i #Dilation size increases with each levels
        #     in_channels = num_inputs if i == 0 else num_channels[i-1]
        #     out_channels = num_channels[i]
        #     layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
        #                              padding=(kernel_size-1) * dilation_size, dropout=dropout, norm_strategy=norm_strategy)]

        for i in range(self.num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            self.temporal_blocks.append(
                TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                              padding=(kernel_size-1) * dilation_size, dropout=dropout, norm_strategy=norm_strategy)
            )

        # self.network = nn.Sequential(*layers)

        self.low_level_conv = nn.Conv1d(in_channels=num_channels[0], out_channels=1, kernel_size=1)
        self.medium_level_conv = nn.Conv1d(in_channels=num_channels[1], out_channels=1, kernel_size=1)
        self.high_level_conv = nn.Conv1d(in_channels=num_channels[2], out_channels=1, kernel_size=1)


    def forward(self, x):
        # output = self.network(x)
        block_features = []
        output = x

        #Get output from each block
        for i in range(self.num_levels):
            output = self.temporal_blocks[i](output)
            block_features.append(output)
        
        low_level_features = block_features[0]
        medium_level_features = block_features[1]
        high_level_features = block_features[2]

        # Apply the 1x1 convolutional layers to reduce the channel dimension for each level
        low_level_reduced = self.low_level_conv(low_level_features)
        medium_level_reduced = self.medium_level_conv(medium_level_features)
        high_level_reduced = self.high_level_conv(high_level_features)

        # Concatenate the reduced features along the channel dimension
        combined_features = torch.cat([low_level_reduced, medium_level_reduced, high_level_reduced], dim=1)

        return combined_features


        
        # return block_features

        # Split the output into low, medium, and high-level features
        low_level_features = output[:, :8, :]
        medium_level_features = output[:, 8:16, :]
        high_level_features = output[:, 16:, :]

        # Apply the 1x1 convolutional layers to reduce the channel dimension for each level
        low_level_reduced = self.low_level_conv(low_level_features)
        medium_level_reduced = self.medium_level_conv(medium_level_features)
        high_level_reduced = self.high_level_conv(high_level_features)

        # Concatenate the reduced features along the channel dimension
        combined_features = torch.cat([low_level_reduced, medium_level_reduced, high_level_reduced], dim=1)

        return combined_features


