import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GraphConv
from torch_geometric.utils import from_networkx
from torch_geometric.data import DataLoader, Data
from preprocessing.GraphEmbedder import GraphEmbedder


class GraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphEncoder, self).__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class GraphDecoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphDecoder, self).__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class GraphAutoencoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels,layers=4,embedding_dim=192):
        super(GraphAutoencoder, self).__init__()
        self.layers = layers
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding()

        self.encoder = GraphEncoder(in_channels, hidden_channels, latent_channels)
        self.decoder = GraphDecoder(latent_channels, hidden_channels, in_channels)

    def forward(self,batch):
        # EMBED INTO a VECTOR of size 31 + 4 * (embed_dim - 1) = 67
        # embedded_batch = self.embedder.embed_into_vector(batch.x)
        # print(embedded_batch.shape, batch.edge_index.shape)
 
        # Perform encoding and decoding
        z = self.encoder(batch.x, batch.edge_index)
        x_hat = self.decoder(z, batch.edge_index)
        return x_hat, z
    
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=False)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)




class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)