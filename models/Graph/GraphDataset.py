import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import GCNConv

# Custom Dataset Class
class MyGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = self.process()

    def create_random_graph(num_nodes, num_features):
        x = torch.randn(num_nodes, num_features)  # Random node features
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))  # Random edges
        return Data(x=x, edge_index=edge_index)

    def process(self):
        data_list = []
        num_graphs = 5  # Example: 5 random graphs
        for _ in range(num_graphs):
            num_nodes = torch.randint(10, 20, (1,)).item()
            num_features = 40
            data = self.create_random_graph(num_nodes, num_features)
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        return data, slices