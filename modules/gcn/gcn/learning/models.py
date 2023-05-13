import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import math
# import time


# GCN Model
class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeatures, nhidden_layers, nclass, dropout, batch_normalization=False):
        super(GCN, self).__init__()

        if batch_normalization:
            self.gc1 = GraphConvolution(nfeatures, nhidden_layers)
            self.bn1 = nn.BatchNorm1d(nhidden_layers)
            self.gc2 = GraphConvolution(nhidden_layers, nclass)
            self.bn2 = nn.BatchNorm1d(nclass)
            self.dropout = dropout
        else:
            self.gc1 = GraphConvolution(nfeatures, nhidden_layers)
            self.gc2 = GraphConvolution(nhidden_layers, nclass)
            self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


# Define the GCN model

class nGCN(torch.nn.Module):
    def __init__(self, hidden_channels, args=None):
        super(nGCN, self).__init__()
        torch.manual_seed(8616)
        self._args = args
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, 32)
        self.conv1 = GCNConv(32, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)

    def forward(self, x, edge_index, device='cpu'):
        x = x.to(device)
        edge_data = edge_index
        A = torch.cat((edge_data[0], edge_data[1]), 0)
        B = torch.cat((edge_data[1], edge_data[0]), 0)
        edge_data = torch.reshape(torch.cat((A, B), 0), (2, -1))
        edge_index = edge_data.to(device)
        x = F.leaky_relu(self.bn1(self.fc1(x)), 0.1)
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.1)
        x = F.leaky_relu(self.bn3(self.fc3(x)), 0.1)
        x = F.leaky_relu(self.conv1(x, edge_index), 0.1)
        x = self.conv2(x, edge_index)
        return x  # Returning logits
    
    def loss(self, output, target, writer=None, index=None):
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(output, target.unsqueeze(1))
        # Logging
        if writer is not None:
            writer.add_scalar("Loss",
                              loss.item(),
                              index)
        return loss
