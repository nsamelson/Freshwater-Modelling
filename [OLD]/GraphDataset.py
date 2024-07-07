import os
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
# from preprocessing.xml2graph import build_graph, convert_to_pyg
from preprocessing.GraphBuilder import GraphBuilder
# from preprocessing.GraphEmbedder import GraphEmbedder
from utils.plot import plot_graph
import xml.etree.ElementTree as ET
import networkx as nx

class GraphDataset(InMemoryDataset):
    def __init__(self, root, equations_file, transform=None, pre_transform=None, pre_filter=None,log=False, force_reload= False, debug=False): #, embedding_dims=10):
        self.equations_file = equations_file
        # self.embedding_dims = embedding_dims
        self.debug = debug
        self.builder = GraphBuilder()
        self.vocab_size = self.builder.vocab_size
        super(GraphDataset, self).__init__(root, transform, pre_transform,pre_filter,log,force_reload)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return [self.equations_file]
    
    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass
    
    def process(self):
        # Read data into huge `Data` list.
        equations_path = os.path.join(self.raw_dir, self.equations_file)
        tree = ET.parse(equations_path)
        root = tree.getroot()
        data_list = []
        for i, formula in enumerate(tqdm(root, desc="Generating Graphs", unit="equations")):
            if self.debug and i>= 4:
                break
            G = self.builder.build_graph(formula)
            G = self.builder.index_texts_in_graph(G)
            if G is None:
                continue
            pyg_graph = self.builder.convert_to_pyg(G)
            data_list.append(pyg_graph)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


