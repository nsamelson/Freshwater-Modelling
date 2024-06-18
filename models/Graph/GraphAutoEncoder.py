import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GraphConv, InnerProductDecoder, BatchNorm
from torch_geometric.utils import from_networkx
from torch_geometric.data import DataLoader, Data
from preprocessing.GraphEmbedder import GraphEmbedder
# from torch.nn import BatchNorm1d



    
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=32, out_channels=16, layers=4,embedding_dim=192,vocab_size=3000,scale_grad_by_freq=False,layer_type=GCNConv, variational=False,batch_norm=False):
        super(Encoder, self).__init__()
        self.layers = layers
        self.embedding_dim = embedding_dim
        self.input_features = in_channels + embedding_dim - 1
        self.embedding = nn.Embedding(vocab_size,embedding_dim,scale_grad_by_freq=scale_grad_by_freq,padding_idx=0)
        self.variational = variational
        self.batch_norm = batch_norm

        # Setup list of Layers
        self.convs = nn.ModuleList()


        # Create as many layers as wanted, minimum # is 2
        for i in range(layers - 1):
            # if first layer, set channels as the number of input features, else put to hidden_channels
            channels = self.input_features if i == 0 else hidden_channels     

            self.convs.append(layer_type(channels,hidden_channels))
            if self.batch_norm:
                self.convs.append(BatchNorm(hidden_channels))
        # self.convs.append(
        #     layer_type(hidden_channels,out_channels)
        # )
        
        # Set output layer
        self.conv_mu = layer_type(hidden_channels, out_channels)

        # If variational (VGAE) add another layer parallel to conv_mu
        if self.variational:
            self.conv_logstd = layer_type(hidden_channels,out_channels)



    def forward(self, x, edge_index):
        indices = x[:,-1]
        one_hot = x[:,:-1]
        embedded = self.embedding(indices)
        new_x = torch.cat((one_hot,embedded),dim=1)

        for i in range(self.layers-1):
            new_x = self.convs[i](new_x, edge_index).relu()
        
        if self.variational:
            return self.conv_mu(new_x, edge_index), self.conv_logstd(new_x,edge_index)
        else:
            return self.conv_mu(new_x, edge_index), None


