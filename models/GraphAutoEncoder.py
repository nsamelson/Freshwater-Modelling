import torch
import torch.nn as nn
import numpy as np
from torch_geometric.utils import from_networkx
from torch_geometric.data import DataLoader

# from preprocessing.GraphEmbedder import MATHML_TAGS,




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