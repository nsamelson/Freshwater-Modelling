
from torch_geometric.nn import GraphSAGE, GCNConv, GraphConv


CONFIG = {
    "lr": 1e-3,
    "num_epochs":1,
    "batch_size":256,
    "layer_type":GCNConv,
    "num_layers":4,
    "out_channels":64,
    "hidden_channels":128,
    "embedding_dims":200,
    "scale_grad_by_freq":True,
    "sample_edges":"sparse",
    "variational":False,
    "batch_norm":False,
    "force_undirected":True,
}