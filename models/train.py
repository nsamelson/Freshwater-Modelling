import json
from matplotlib import pyplot as plt
import numpy as np
# from sklearn.base import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset, collate, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import xml.etree.ElementTree as ET

from tqdm import tqdm
from preprocessing.GraphEmbedder import GraphEmbedder, MATHML_TAGS
from models.Graph.GraphDataset import GraphDataset
from models.Graph.GraphAutoEncoder import GraphAutoencoder
import random



def main():
    # Fix the randomness
    seed_value = 0
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    print("Loading dataset...")
    dataset_path = "dataset/graph_formulas_katex_debug.pt"
    dataset = torch.load(dataset_path)

    embedder = GraphEmbedder(scale_by_freq=False)

    # def collate_fn(data_list):
    #     batch = Batch.from_data_list(data_list)
    #     batch.x = embedder.embed_into_vector(batch.x)
    #     return batch

    # Create a DataLoader
    batch_size = 3
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=0)
    dataloader = EmbeddedDataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, embedder=embedder)



    in_channels = len(MATHML_TAGS) + 4 * (embedder.embedding_dims - 1) 

    model = GraphAutoencoder(in_channels,16,8,embedder,MATHML_TAGS)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            x_hat, z = model(batch)
            loss = criterion(x_hat, batch.x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

    # Example of using the model for inference
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x_hat, z = model(batch)
            print("Original Node Features:", batch.x)
            print("Reconstructed Node Features:", x_hat)
            # print("what is z: ",z)
            # Calculate reconstruction error (e.g., Mean Squared Error)
            reconstruction_error = np.mean((batch.x.numpy() - x_hat.detach().numpy()) ** 2)
            print("Reconstruction Error:", reconstruction_error)


    
class EmbeddedDataLoader(DataLoader):
    def __init__(self, *args, embedder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedder = embedder

    def __iter__(self):
        for batch in super().__iter__():
            # Apply the embedding function to the batch
            batch.x = self.embedder.embed_into_vector(batch.x)
            yield batch