import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GraphConv, InnerProductDecoder, BatchNorm, VGAE, MessagePassing
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

METHODS = {
    "onehot": ["concat","tag","pos"],
    "embed": ["tag","concat","combined","mi","mo","mtext","mn"],
    "linear":["mn"],
    "scale": ["log"],
}


class GraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=32, out_channels=16, layers:int=4,layer_type=GCNConv, batch_norm=False):
        super(GraphEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None

        # Create as many layers as wanted, minimum # is 2
        for i in range(layers - 1):
            channels = in_channels if i == 0 else hidden_channels
            self.convs.append(layer_type(channels,hidden_channels))
            if batch_norm:
                self.batch_norms.append(BatchNorm(hidden_channels))
        
        # Last 2 layers in parallel
        self.conv_mu = layer_type(hidden_channels, out_channels)
        self.conv_logstd = layer_type(hidden_channels,out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        for i, layer in enumerate(self.convs):
            x = layer(x, edge_index, edge_weight).relu()  # GCN
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)  # Batch Norm
        
        mu = self.conv_mu(x, edge_index, edge_weight)
        logstd = self.conv_logstd(x, edge_index, edge_weight)

        return mu, logstd
    
class GraphDecoder(torch.nn.Module):
    def __init__(self, out_channels, hidden_channels=32, in_channels=16, layers=4, layer_type=GCNConv, edge_features_dim=1, batch_norm=False):
        super(GraphDecoder, self).__init__()
        self.edge_features_dim = edge_features_dim
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        self.edge_decoder = InnerProductDecoder()
        
        # Create as many layers as the encoder, mirroring the structure
        for i in range(layers - 1):
            channels = in_channels if i == 0 else hidden_channels
            self.convs.append(layer_type(channels, hidden_channels))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Last layer to reconstruct the original input feature dimensions
        self.conv_out = layer_type(hidden_channels, out_channels)

        # Linear layer to reconstruct edge features
        self.edge_features_recon = nn.Linear(in_channels*2, edge_features_dim) # 1 because binary thing

    def forward(self, *args,**kwargs):
        return self.edge_decoder(*args,**kwargs) 

    
    def node_decoder(self, z, edge_index, edge_weight=None):
        for i, layer in enumerate(self.convs):
            z = layer(z, edge_index, edge_weight).relu()  # GCN
            if self.batch_norms is not None:
                z = self.batch_norms[i](z)  # Batch Norm
        
        return self.conv_out(z, edge_index)
    
    def edge_features_decoder(self,z,edge_index):
        # Get the source and target node embeddings for each edge
        row, col = edge_index
        edge_features = torch.cat([z[row], z[col]], dim=1)
        
        # Reconstruct edge features
        edge_features_recon = torch.sigmoid(self.edge_features_recon(edge_features)).squeeze()
        return edge_features_recon




class GraphVAE(VGAE):
    def __init__(self, encoder: GraphEncoder, decoder: GraphDecoder, num_embeddings:int | dict, embedding_dim, embedding_method, scale_grad_by_freq):
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
        self.scale_grad = scale_grad_by_freq
        self.unknown_id = 1

        # if embedding_method not in EMBEDDINGS:
        #     raise ValueError(f"Invalid embedding method. Expected one of {EMBEDDINGS}, but got {embedding_method}")

        # Generate embeddings
        self.initialize_embeddings()
        

    def initialize_embeddings(self):
        """
        Initializes embeddings based on the specified embedding method.
        """
        if not hasattr(self, 'embeddings'):
            self.embeddings = {}

            if "onehot" in self.embedding_method and len(self.embedding_method["onehot"]) != 0:
                for vocab in self.embedding_method["onehot"]:
                    if vocab == "tag":
                        input_dims = len(MATHML_TAGS)
                    elif vocab == "pos":
                        input_dims = 256
                    else:
                        input_dims = self.embedding_dim

                    self.embeddings[vocab] = nn.Embedding(input_dims, input_dims)
                    self.embeddings[vocab].weight.data = torch.eye(input_dims)

            if "embed" in self.embedding_method and len(self.embedding_method["embed"]) != 0:
                for vocab in self.embedding_method["embed"]:
                    if vocab == "tag":
                        input_dims = len(MATHML_TAGS) 
                    elif vocab in ["concat","combined"]:
                        input_dims = self.num_embeddings
                    elif vocab in ["mi","mn","mtext","mo"]:
                        input_dims = self.num_embeddings[vocab]
                    else:
                        raise ValueError(f"Invalid vocab selected. Expected one of {METHODS['embed']}, but got {vocab}")
                    self.embeddings[vocab] = nn.Embedding(input_dims, self.embedding_dim, scale_grad_by_freq=self.scale_grad, padding_idx=0)

            if "linear" in self.embedding_method and len(self.embedding_method["linear"]) != 0:
                for vocab in self.embedding_method["linear"]:
                    self.embeddings[vocab] = nn.Linear(1,self.embedding_dim)



    def embed_x(self,x:Tensor, tag_index:Tensor, pos:Tensor, nums:Tensor) -> Tensor:
        """
        Embeds the input index based on the specified embedding method and concatenates
        it with other features to form a new vector.

        Args:
            x (Tensor): The input tensor to embed.
            tag_index (Tensor): The tag indices corresponding to the input tensor.
            pos (Tensor): Position indices.
            num (Tensor): Numerical values to embed.

        Returns:
            Tensor: The embedded and concatenated feature vector.
        """
        embedded = []

        if "onehot" in self.embedding_method and len(self.embedding_method["onehot"]) != 0:
            for vocab in self.embedding_method["onehot"]:
                if vocab == "tag":
                    embedded.append(self.embeddings["tag"](tag_index))
                elif vocab == "concat":
                    x = torch.where(x >= self.embedding_dim, torch.tensor(self.unknown_id, device=x.device), x)
                    embedded.append(self.embeddings["concat"](x))
                elif vocab == "pos":
                    embedded.append(self.embeddings["pos"](pos))
                else:
                    raise ValueError(f"Invalid one-hot vocab selected. Expected one of {METHODS['onehot']}, but got {vocab}")

        if "embed" in self.embedding_method and len(self.embedding_method["embed"]) != 0:
            for vocab in self.embedding_method["embed"]:
                if vocab == "tag":
                    embedded.append(self.embeddings["tag"](tag_index))
                elif vocab in ["concat","combined"]:
                    embedded.append(self.embeddings[vocab](x))
                elif vocab in ["mi","mn","mtext","mo"]:
                    mask = (tag_index == MATHML_TAGS.index(vocab))
                    vector = torch.zeros(x.size(0),self.embedding_dim, dtype=torch.float32, device=x.device) #.unsqueeze(1).repeat(1, self.embedding_dim)
                    vector[mask] = self.embeddings[vocab](x[mask])
                    embedded.append(vector)
                else:
                    raise ValueError(f"Invalid embed vocab selected. Expected one of {METHODS['embed']}, but got {vocab}")

        if "linear" in self.embedding_method and len(self.embedding_method["linear"]) != 0:
                for vocab in self.embedding_method["linear"]:
                    mask = (nums != -1)
                    nums[mask] = self.feature_scale(nums[mask])
                    vector = torch.zeros(x.size(0), self.embedding_dim, dtype=torch.float32, device=x.device)
                    vector[mask] = self.embeddings[vocab](nums[mask].unsqueeze(1))
                    embedded.append(vector)


        new_x = torch.cat(embedded,dim=1)
        return new_x
    
    def feature_scale(self, nums):
        scaling = self.embedding_method["scale"]

        if scaling in METHODS["scale"]:
            if scaling == "log":
                return torch.log10(torch.clamp(nums, min=1e-6) + 1e-7)
        else:
            raise ValueError(f"Invalid feature scaling. Expected one of {METHODS['scale']}, but got {scaling}")
    
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

        ef_recon = self.decoder.edge_features_decoder(z,edge_index)

        return x_recon, e_recon, ef_recon
    
    def recon_full_loss(self, z, x, pos_edge_index, neg_edge_index, edge_weight = None, alpha = 1, beta = 0, gamma = 0):
        
        # Compute link loss
        adj_loss = self.recon_loss(z, pos_edge_index,neg_edge_index)

        # Node features loss
        decoded_x = self.decoder.node_decoder(z, pos_edge_index)
        feature_loss = F.cross_entropy(decoded_x,x)

        # Edge features loss
        ef_loss = 0
        if edge_weight is not None:
            decoded_ef = self.decoder.edge_features_decoder(z, pos_edge_index)
            
            if self.decoder.edge_features_dim == 1:
                ef_loss = F.binary_cross_entropy_with_logits(decoded_ef,edge_weight.float())
            else:
                ef_loss = F.cross_entropy(decoded_ef,edge_weight.float())

        return alpha * adj_loss + beta * feature_loss + gamma * ef_loss


    def calculate_accuracy(self, z, x, pos_edge_index, edge_weight=None):

        # Edge index accuracy
        e_recon = self.decoder(z, pos_edge_index,True)
        e_recon = (e_recon > 0.5).float()
        correct_e = (e_recon == 1).sum().item()
        edge_accuracy = correct_e / pos_edge_index.size(1)

        # Node features accuracy
        decoded_x = self.decoder.node_decoder(z, pos_edge_index)
        _, predicted_nodes = torch.max(decoded_x, dim=1)
        _, true_nodes = torch.max(x,dim=1)

        correct_nodes = (predicted_nodes == true_nodes).sum().item()
        node_accuracy = correct_nodes / true_nodes.size(0)

        # Edge features accuracy
        edge_features_accuracy = None
        if edge_weight is not None:
            decoded_ef = self.decoder.edge_features_decoder(z, pos_edge_index)
            if self.decoder.edge_features_dim == 1:
                predicted_edges = (torch.sigmoid(decoded_ef) > 0.5).float()
                correct_edges = (predicted_edges == edge_weight).sum().item()
                edge_features_accuracy = correct_edges / edge_weight.size(0)
            else:
                _, predicted_edges = torch.max(decoded_ef, dim=1)
                correct_edges = (predicted_edges == edge_weight).sum().item()
                edge_features_accuracy = correct_edges / edge_weight.size(0)

        return node_accuracy, edge_accuracy, edge_features_accuracy

