import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GraphConv, InnerProductDecoder, BatchNorm, VGAE
from torch_geometric.utils import from_networkx
from torch_geometric.data import DataLoader, Data
# from torch.nn import BatchNorm1d
from torch.nn import Module
from torch import Tensor

from config import MATHML_TAGS

# MAX_LOGSTD = 10
# EPS = 1e-15

EMBEDDINGS = [
    "OHE_Concat",
    "Embed_Concat",
    "OHE_Tags_Embed_Combined",
    "OHE_Tags_Embed_Split",
    "MultiEmbed_Split",
]

class GraphVAE(VGAE):
    def __init__(self, encoder: Module, decoder: Module, num_embeddings:int | dict, embedding_dim, embedding_method, scale_grad_by_freq):
        """
        Initializes the GraphVAE model.

        Args:
            encoder (Module): The encoder module.
            decoder (Module): The decoder module.
            embedding_dim (int): The dimension of the embeddings.
            num_embeddings (int): The number of embeddings.
            embedding_method (str): The method for embedding. Must be one of ["OHE_Concat", "Embed_Concat", "OHE_Tags_Embed_Combined", 
                "OHE_Tags_Embed_Split", "MultiEmbed_Split"].
            scale_grad_by_freq (bool): Whether to scale the gradient by the frequency of the words.
        
        Raises:
            ValueError: If embedding_method is not one of the specified options.
        """
        super(GraphVAE, self).__init__(encoder, decoder)
        self.num_embeddings = num_embeddings
        self.embedding_method = embedding_method
        self.embedding_dim = embedding_dim
        self.scale_grad_by_freq = scale_grad_by_freq
        # self.text_embedding = nn.Embedding(num_embeddings, embedding_dim, scale_grad_by_freq= scale_grad_by_freq, padding_idx=0)
        # self.tag_embedding = nn.Embedding(num_embeddings, embedding_dim, scale_grad_by_freq= scale_grad_by_freq, padding_idx=0)
        self.unknown_id = 1

        if embedding_method not in EMBEDDINGS:
            raise ValueError(f"Invalid embedding method. Expected one of {EMBEDDINGS}, but got {embedding_method}")

    def embed_x(self,x, tag_index, pos=0, onehot=False) -> Tensor:
        """
        Embeds the index and concats to a new vector
        """
        if self.embedding_method == "OHE_Concat":
            self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
            self.embedding.weight.data = torch.eye(self.num_embeddings)
            # x = self.unknown_id if x >= self.num_embeddings else x
            x = torch.where(x >= self.num_embeddings, torch.tensor(self.unknown_id, device=x.device), x)
            new_x = self.embedding(x)

        elif self.embedding_method == "Embed_Concat":
            self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim, scale_grad_by_freq= self.scale_grad_by_freq, padding_idx=0)
            new_x = self.embedding(x)

        elif self.embedding_method == "OHE_Tags_Embed_Combined":
            self.tag_embedding = nn.Embedding(len(MATHML_TAGS), len(MATHML_TAGS))
            self.tag_embedding.weight.data = torch.eye(len(MATHML_TAGS))
            embedded_tag = self.tag_embedding(tag_index)

            self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim, scale_grad_by_freq= self.scale_grad_by_freq, padding_idx=0)
            embedded_x = self.embedding(x)

            new_x = torch.cat((embedded_tag,embedded_x),dim=1)

        elif self.embedding_method == "OHE_Tags_Embed_Split":
            self.tag_embedding = nn.Embedding(len(MATHML_TAGS), len(MATHML_TAGS))
            self.tag_embedding.weight.data = torch.eye(len(MATHML_TAGS))
            embedded_tag = self.tag_embedding(tag_index)


            self.mi_embedding = nn.Embedding(self.num_embeddings["mi"], self.embedding_dim, scale_grad_by_freq= self.scale_grad_by_freq, padding_idx=0)
            self.mo_embedding = nn.Embedding(self.num_embeddings["mo"], self.embedding_dim, scale_grad_by_freq= self.scale_grad_by_freq, padding_idx=0)
            self.mtext_embedding = nn.Embedding(self.num_embeddings["mtext"], self.embedding_dim, scale_grad_by_freq= self.scale_grad_by_freq, padding_idx=0)
            self.mn_embedding = nn.Embedding(self.num_embeddings["mn"], self.embedding_dim, scale_grad_by_freq= self.scale_grad_by_freq, padding_idx=0)

            # Create masks
            mi_mask = (tag_index == MATHML_TAGS.index("mi"))
            mo_mask = (tag_index == MATHML_TAGS.index("mo"))
            mtext_mask = (tag_index == MATHML_TAGS.index("mtext"))
            mn_mask = (tag_index == MATHML_TAGS.index("mn"))

            # Initialize embeddings with zeros
            embedded_mi = torch.zeros_like(x, dtype=torch.float32, device=x.device).unsqueeze(1).repeat(1, self.embedding_dim)
            embedded_mo = torch.zeros_like(x, dtype=torch.float32, device=x.device).unsqueeze(1).repeat(1, self.embedding_dim)
            embedded_mtext = torch.zeros_like(x, dtype=torch.float32, device=x.device).unsqueeze(1).repeat(1, self.embedding_dim)
            embedded_mn = torch.zeros_like(x, dtype=torch.float32, device=x.device).unsqueeze(1).repeat(1, self.embedding_dim)


            # Apply embeddings conditionally
            embedded_mi[mi_mask] = self.mi_embedding(x[mi_mask])
            embedded_mo[mo_mask] = self.mo_embedding(x[mo_mask])
            embedded_mtext[mtext_mask] = self.mtext_embedding(x[mtext_mask])
            embedded_mn[mn_mask] = self.mn_embedding(x[mn_mask])

            new_x = torch.cat((embedded_tag, embedded_mi, embedded_mo, embedded_mtext, embedded_mn), dim=1)

        elif self.embedding_method == "MultiEmbed_Split":
            self.tag_embedding = nn.Embedding(len(MATHML_TAGS), self.embedding_dim, scale_grad_by_freq= self.scale_grad_by_freq, padding_idx=0)
            embedded_tag = self.tag_embedding(tag_index)

            self.mi_embedding = nn.Embedding(self.num_embeddings["mi"], self.embedding_dim, scale_grad_by_freq= self.scale_grad_by_freq, padding_idx=0)
            self.mo_embedding = nn.Embedding(self.num_embeddings["mo"], self.embedding_dim, scale_grad_by_freq= self.scale_grad_by_freq, padding_idx=0)
            self.mtext_embedding = nn.Embedding(self.num_embeddings["mtext"], self.embedding_dim, scale_grad_by_freq= self.scale_grad_by_freq, padding_idx=0)
            self.mn_embedding = nn.Embedding(self.num_embeddings["mn"], self.embedding_dim, scale_grad_by_freq= self.scale_grad_by_freq, padding_idx=0)

            # Create masks
            mi_mask = (tag_index == MATHML_TAGS.index("mi"))
            mo_mask = (tag_index == MATHML_TAGS.index("mo"))
            mtext_mask = (tag_index == MATHML_TAGS.index("mtext"))
            mn_mask = (tag_index == MATHML_TAGS.index("mn"))

            # Initialize embeddings with zeros
            embedded_mi = torch.zeros_like(x, dtype=torch.float32, device=x.device).unsqueeze(1).repeat(1, self.embedding_dim)
            embedded_mo = torch.zeros_like(x, dtype=torch.float32, device=x.device).unsqueeze(1).repeat(1, self.embedding_dim)
            embedded_mtext = torch.zeros_like(x, dtype=torch.float32, device=x.device).unsqueeze(1).repeat(1, self.embedding_dim)
            embedded_mn = torch.zeros_like(x, dtype=torch.float32, device=x.device).unsqueeze(1).repeat(1, self.embedding_dim)


            # Apply embeddings conditionally
            embedded_mi[mi_mask] = self.mi_embedding(x[mi_mask])
            embedded_mo[mo_mask] = self.mo_embedding(x[mo_mask])
            embedded_mtext[mtext_mask] = self.mtext_embedding(x[mtext_mask])
            embedded_mn[mn_mask] = self.mn_embedding(x[mn_mask])

            new_x = torch.stack([embedded_tag, embedded_mi, embedded_mo, embedded_mtext, embedded_mn], dim=0)

        # indices = x[:,-1].long() # making sure it's an integer
        # one_hot = x[:,:-1]
        # embedded = self.embedding(indices)
        # new_x = torch.cat((one_hot,embedded),dim=1)
        return new_x
    
    def reverse_embedding(self, x_recon):
        """
        Reverses the embedding process to find the index
        """
        embedded_part = x_recon[:, -self.embedding.embedding_dim:]
        one_hot = x_recon[:, :-self.embedding.embedding_dim]

        # Binary output to get the one-hot back
        # one_hot = (torch.sigmoid(one_hot) > 0.5).float()
        one_hot = (one_hot == one_hot.max(dim=-1, keepdim=True)[0]).float()
        # one_hot = torch.zeros(one_hot.shape)

        # Find the index back
        cosine_sim = F.cosine_similarity(embedded_part.unsqueeze(1), self.embedding.weight.unsqueeze(0), dim=2)
        decoded_indices = cosine_sim.argmax(dim=1)

        # Combine the one-hot with the index
        x_combined = torch.cat((one_hot, decoded_indices.unsqueeze(1).float()), dim=1).long()
        
        return x_combined
    
    def decode_all(self,z, edge_index, sigmoid=True):
        e_recon = self.decoder(z, edge_index,sigmoid)
        e_recon = (e_recon > 0.5).float()

        x_recon = self.decoder.node_decoder(z, edge_index)
        x_recon = self.reverse_embedding(x_recon)

        # ef_recon = None # TODO: decode the edge features

        return x_recon, e_recon
    
    def recon_full_loss(self, z, x, pos_edge_index, neg_edge_index, alpha = 1, beta = 0, gamma = 0):
        
        # Compute link loss
        adj_loss = self.recon_loss(z, pos_edge_index,neg_edge_index)

        # Node features loss
        feature_loss = F.mse_loss(x, self.decoder.node_decoder(z, pos_edge_index))

        # Edge features loss
        # edge_feat_loss = 0 # TODO: compute edge feature loss if not None

        return alpha * adj_loss + beta * feature_loss


   

class GraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=32, out_channels=16, layers:int=4,layer_type=GCNConv):
        super(GraphEncoder, self).__init__()
        self.convs = nn.ModuleList()

        # Create as many layers as wanted, minimum # is 2
        for i in range(layers - 1):
            channels = in_channels if i == 0 else hidden_channels
            self.convs.append(layer_type(channels,hidden_channels))
        
        # Last 2 layers in parallel
        self.conv_mu = layer_type(hidden_channels, out_channels)
        self.conv_logstd = layer_type(hidden_channels,out_channels)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)

        return mu, logstd
    
class GraphDecoder(torch.nn.Module):
    def __init__(self, out_channels, hidden_channels=32, in_channels=16, layers=4, layer_type=GCNConv):
        super(GraphDecoder, self).__init__()
        self.convs = nn.ModuleList()
        self.edge_decoder = InnerProductDecoder()
        
        # Create as many layers as the encoder, mirroring the structure
        for i in range(layers - 1):
            channels = in_channels if i == 0 else hidden_channels
            self.convs.append(layer_type(channels, hidden_channels))
        
        # Last layer to reconstruct the original input feature dimensions
        self.conv_out = layer_type(hidden_channels, out_channels)


    def forward(self, *args,**kwargs):
        return self.edge_decoder(*args,**kwargs) 

    
    def node_decoder(self, z, edge_index):
        for conv in self.convs:
            z = conv(z, edge_index).relu()
        
        return self.conv_out(z, edge_index)

    # def forward(self, z, edge_index, sigmoid:bool = True): # *args,**kwargs
    #     for conv in self.convs:
    #         z = conv(z, edge_index).relu()
        
    #     x_recon = self.conv_out(z, edge_index)
    #     e_recon = self.edge_decoder(z,edge_index,sigmoid)
    #     return x_recon, e_recon
        
    
# class Encoder(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels=32, out_channels=16, layers=4,embedding_dim=192,vocab_size=3000,scale_grad_by_freq=False,layer_type=GCNConv, variational=False,batch_norm=False):
#         super(Encoder, self).__init__()
#         self.layers = layers
#         self.embedding_dim = embedding_dim
#         self.input_features = in_channels + embedding_dim - 1
#         self.embedding = nn.Embedding(vocab_size,embedding_dim,scale_grad_by_freq=scale_grad_by_freq,padding_idx=0)
#         self.variational = variational
#         # self.batch_norm = batch_norm

#         # Setup list of Layers
#         self.convs = nn.ModuleList()

#         # Create as many layers as wanted, minimum # is 2
#         for i in range(layers - 1):
#             # if first layer, set channels as the number of input features, else put to hidden_channels
#             channels = self.input_features if i == 0 else hidden_channels
#             self.convs.append(layer_type(channels,hidden_channels))
#             # if self.batch_norm and i < layers - 2:  # Add BN after each but the last layer
#             #     self.convs.append(nn.BatchNorm1d(hidden_channels))
        
#         # Last 2 layers in parallel
#         self.conv_mu = layer_type(hidden_channels, out_channels)
#         self.conv_logstd = layer_type(hidden_channels,out_channels)



#     def forward(self, x, edge_index):
#         # x, edge_index = batch.x, batch.edge_index

#         indices = x[:,-1]
#         one_hot = x[:,:-1]
#         embedded = self.embedding(indices)
#         new_x = torch.cat((one_hot,embedded),dim=1)

#         for conv in self.convs:
#             new_x = conv(new_x, edge_index).relu()
        
#         mu = self.conv_mu(new_x, edge_index)
#         logstd = self.conv_logstd(new_x, edge_index)

    
#         if self.variational:
#             return mu, logstd
#         else:
#             return mu
        
    # def find_nearest_embeddings(self, embedded_vectors):
    #     embedding_matrix = self.embedding.weight.data  # [vocab_size, embedding_dim]
    #     embedded_vectors = embedded_vectors.cpu()  # [num_nodes, embedding_dim]
    #     cosine_sim = F.cosine_similarity(embedded_vectors.unsqueeze(1), embedding_matrix.unsqueeze(0), dim=2)
    #     indices = cosine_sim.argmax(dim=1)
    #     return indices


# class Decoder(torch.nn.Module):
#     def __init__(self, ):
#         super(Decoder, self).__init__()
#         # self.embedding_weight = embedding_weight

#     def decode_edges(self, z):
#         return torch.sigmoid(torch.matmul(z, z.t()))  # Inner product for edge reconstruction

#     def decode_node_features(self, z):
#         cosine_sim = F.cosine_similarity(z.unsqueeze(1), self.embedding_weight.unsqueeze(0), dim=2)
#         decoded_indices = cosine_sim.argmax(dim=1)
#         return decoded_indices

#     def reconstruct_node_features(self, one_hot, decoded_indices):
#         combined_features = torch.cat((one_hot, decoded_indices.unsqueeze(1).float()), dim=1)
#         return combined_features