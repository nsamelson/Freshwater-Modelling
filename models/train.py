import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import GCNConv
import xml.etree.ElementTree as ET
from preprocessing.GraphEmbedder import GraphEmbedder


def main():
    print("Loading dataset...")
    dataset = torch.load("dataset/graph_formulas_katex_debug.pt")
    print("Finished loading dataset!")
    print(dataset)
    # TODO: build dataset with GraphDataset Class
    # TODO: import model
    # TODO: create train loop (separate function)
    # TODO: poof we have a trained model