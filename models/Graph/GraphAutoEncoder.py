import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GraphConv, InnerProductDecoder
from torch_geometric.utils import from_networkx
from torch_geometric.data import DataLoader, Data
from preprocessing.GraphEmbedder import GraphEmbedder




    
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, layers=4,embedding_dim=192,vocab_size=3000,scale_grad_by_freq=False,layer_type=GCNConv):
        super(Encoder, self).__init__()
        self.layers = layers
        self.embedding_dim = embedding_dim
        self.input_features = in_channels + embedding_dim - 1
        self.embedding = nn.Embedding(vocab_size,embedding_dim,scale_grad_by_freq=scale_grad_by_freq,padding_idx=0)


        # Create as many layers as possible, minimum #=2
        self.convs = nn.ModuleList()
        self.convs.append(
            layer_type(self.input_features,hidden_channels)
        )
        for _ in range(layers - 2):
            self.convs.append(
                layer_type(hidden_channels,hidden_channels)
            )
        self.convs.append(
            layer_type(hidden_channels,out_channels)
        )


    def forward(self, x, edge_index):
        indices = x[:,-1]
        one_hot = x[:,:-1]
        embedded = self.embedding(indices)
        new_x = torch.cat((one_hot,embedded),dim=1)

        for i in range(self.layers-1):
            new_x = F.relu(self.convs[i](new_x, edge_index))
        return self.convs[-1](new_x, edge_index)



class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)