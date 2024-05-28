import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import GCNConv

from preprocessing.GraphEmbedder import GraphEmbedder

# Custom Dataset Class
class GraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,pre_filter=None,embedder = GraphEmbedder):
        super(GraphDataset, self).__init__(root, transform, pre_transform,pre_filter)
        data_list = torch.load(root)
        self.embedder = embedder
        self.data, self.slices = self.collate(data_list)
    

    
    # @staticmethod
    def collate(self,data_list):
        data_list.x = self.embedder.embed_into_vector(data_list.x)
        data, slices = Data.collate(data_list)
        return data, slices

    # def create_random_graph(num_nodes, num_features):
    #     x = torch.randn(num_nodes, num_features)  # Random node features
    #     edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))  # Random edges
    #     return Data(x=x, edge_index=edge_index)

    # def process(self):
    #     # data_list = []
    #     # num_graphs = 5  # Example: 5 random graphs
    #     # for _ in range(num_graphs):
    #     #     num_nodes = torch.randint(10, 20, (1,)).item()
    #     #     num_features = 40
    #     #     data = self.create_random_graph(num_nodes, num_features)
    #     #     data_list.append(data)
        
    #     data, slices = self.collate(self.root)
    #     return data, slices

# class MyOwnDataset(InMemoryDataset):
#     def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
#         super().__init__(root, transform, pre_transform, pre_filter)
#         self.load(self.processed_paths[0])


#     @property
#     def raw_file_names(self):
#         return ['some_file_1', 'some_file_2', ...]

#     @property
#     def processed_file_names(self):
#         return ['data.pt']

#     # def download(self):
#     #     # Download to `self.raw_dir`.
#     #     download_url(url, self.raw_dir)

#     def process(self):
#         # Read data into huge `Data` list.
#         data_list = [...]

#         if self.pre_filter is not None:
#             data_list = [data for data in data_list if self.pre_filter(data)]

#         if self.pre_transform is not None:
#             data_list = [self.pre_transform(data) for data in data_list]

#         self.save(data_list, self.processed_paths[0])
