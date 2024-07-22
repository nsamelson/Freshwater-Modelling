
from torch_geometric.nn import GraphSAGE, GCNConv, GraphConv



ROOT_DIR = "/data/nsam947/Freshwater-Modelling/"

CONFIG = {
    "lr": 1e-3,
    "num_epochs":1,
    "batch_size":256,
    "layer_type":GCNConv,
    "num_layers":4,
    "out_channels":32,
    "hidden_channels":64,
    "scale_grad_by_freq":True,
    "variational":True,
    "batch_norm":False,
    "force_undirected":True,
    "alpha":1,
    "beta":1,
    "gamma":1,
    # "method":{
    #     "onehot": {},
    #     "embed": {"concat": 256},
    #     "linear": {},
    #     "scale": "log",
    #     "loss":"mse"
    # },
    "shuffle": False,
    "max_num_nodes": 100,
    "latex_set":"OleehyO",
    "vocab_type":"concat",
    "xml_name":"default",
    "force_reload": False,
    "sample_edges":"sparse",
    "gen_sparse_edges": True,
    "train_edge_features": False,
}

MATHML_TAGS = [
    "maction",
    "math",
    "menclose",
    "merror", 
    "mfenced",
    "mfrac", 
    "mglyph", 
    "mi", 	
    "mlabeledtr", 
    "mmultiscripts", 
    "mn",
    "mo",
    "mover", 	
    "mpadded", 	
    "mphantom", 	
    "mroot", 	
    "mrow", 
    "ms", 	
    "mspace",
    "msqrt",
    "mstyle",
    "msub",
    "msubsup",  
    "msup",
    "mtable",
    "mtd",
    "mtext",
    "mtr",
    "munder",
    "munderover",
    "semantics", 
]