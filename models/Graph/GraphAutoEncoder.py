import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from torch_geometric.data import DataLoader, Data

# from preprocessing.GraphEmbedder import MATHML_TAGS,

class GraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class GraphDecoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphDecoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class GraphAutoencoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels):
        super(GraphAutoencoder, self).__init__()
        self.encoder = GraphEncoder(in_channels, hidden_channels, latent_channels)
        self.decoder = GraphDecoder(latent_channels, hidden_channels, in_channels)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        x_hat = self.decoder(z, edge_index)
        return x_hat, z
    


class YourModel(nn.Module):
    def __init__(self, embedder,mathml_tags):
        super(YourModel, self).__init__()
        self.embedder = embedder
        self.mathml_tags = mathml_tags
        # Define other layers of your model here

    def forward(self, batch):
        batch_indices = batch.x
        embedded_batch = []
        for tag in ["mi", "mo", "mtext"]:
            tag_indices = batch_indices[:, self.mathml_tags.index(tag)]
            embedded_tag = self.embedder.embeddings[tag](tag_indices)
            embedded_batch.append(embedded_tag)
        # Concatenate embeddings for all tags and other features
        embedded_batch = torch.cat(embedded_batch, dim=1)
        # Pass through other layers of your model
        # output = self.some_layers(embedded_batch)
        # return output