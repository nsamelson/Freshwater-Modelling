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
        # self.batch_norm = batch_norm

        # Setup list of Layers
        self.convs = nn.ModuleList()

        # Create as many layers as wanted, minimum # is 2
        for i in range(layers - 1):
            # if first layer, set channels as the number of input features, else put to hidden_channels
            channels = self.input_features if i == 0 else hidden_channels
            self.convs.append(layer_type(channels,hidden_channels))
            # if self.batch_norm and i < layers - 2:  # Add BN after each but the last layer
            #     self.convs.append(nn.BatchNorm1d(hidden_channels))
        
        # Last 2 layers in parallel
        self.conv_mu = layer_type(hidden_channels, out_channels)
        self.conv_logstd = layer_type(hidden_channels,out_channels)



    def forward(self, x, edge_index):
        # x, edge_index = batch.x, batch.edge_index

        indices = x[:,-1]
        one_hot = x[:,:-1]
        embedded = self.embedding(indices)
        new_x = torch.cat((one_hot,embedded),dim=1)

        for conv in self.convs:
            new_x = conv(new_x, edge_index).relu()
        
        mu = self.conv_mu(new_x, edge_index)
        logstd = self.conv_logstd(new_x, edge_index)

    
        if self.variational:
            return mu, logstd
        else:
            return mu
        
    # def find_nearest_embeddings(self, embedded_vectors):
    #     embedding_matrix = self.embedding.weight.data  # [vocab_size, embedding_dim]
    #     embedded_vectors = embedded_vectors.cpu()  # [num_nodes, embedding_dim]
    #     cosine_sim = F.cosine_similarity(embedded_vectors.unsqueeze(1), embedding_matrix.unsqueeze(0), dim=2)
    #     indices = cosine_sim.argmax(dim=1)
    #     return indices


class Decoder(torch.nn.Module):
    def __init__(self, embedding_weight):
        super(Decoder, self).__init__()
        self.embedding_weight = embedding_weight

    def decode_edges(self, z):
        return torch.sigmoid(torch.matmul(z, z.t()))  # Inner product for edge reconstruction

    def decode_node_features(self, z):
        cosine_sim = F.cosine_similarity(z.unsqueeze(1), self.embedding_weight.unsqueeze(0), dim=2)
        decoded_indices = cosine_sim.argmax(dim=1)
        return decoded_indices

    def reconstruct_node_features(self, one_hot, decoded_indices):
        combined_features = torch.cat((one_hot, decoded_indices.unsqueeze(1).float()), dim=1)
        return combined_features