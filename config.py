
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
    "sample_edges":"sparse",
    "variational":False,
    "batch_norm":False,
    "force_undirected":True,
    "alpha":1,
    "beta":1,
    "gamma":1,
    "method":{
        "onehot": {"concat": 256},
        "embed": {},
        "linear": {},
        "scale": "log",
        "loss":"cross_entropy"
    },
    "shuffle": False
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