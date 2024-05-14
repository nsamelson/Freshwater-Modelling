import json
import math
import re
import xml.etree.ElementTree as ET
import networkx as nx
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.data import DataLoader, Data
from tqdm import tqdm
# from utils.plot import plot_labels_frequency
from utils import save, stats, plot
# import torch_geometric.nn as pyg_nn
# import torch_geometric.utils as pyg_utils 
# import torch_geometric.transforms as T
# import torch
# from tensorboardX import SummaryWriter
# from sklearn.manifold import TSNE

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

def main(generate_stats=False, debug=True):
    """
    Generate a graph from mathml XML equations. 
    Source : https://github.com/Whadup/arxiv_learning/blob/ecml/arxiv_learning/data/load_mathml.py
    """
    

    # Load the MathML equation 
    tree = ET.parse('out/test.xml')
    # tree = ET.parse('dataset/cleaned_formulas.xml')
    root = tree.getroot()

    graphs = []


    # Iterate through each XML equation and create a graph
    for i, formula in enumerate(tqdm(root,desc="Generating Graphs",unit="equations")):

        if debug and i >= 10:
            break
        
        # Build graph and add to list
        G = build_graph(formula)
        graphs.append(G)

    G = graphs[0]
    # nx.write_weighted_edgelist(G,"out/graph.txt")
    print(G)
    print(G.nodes(data=True))
    print(G.edges())
        
    # TODO: DO I NEED TO ENCODE?
    # CHATGPT : In PyTorch Geometric (PyG), node features are typically numerical values stored in a tensor.One common approach is to use encoding techniques such as one-hot encoding, label encoding, or even embedding vectors.
    #   - Label Encoding for tag: Assign a unique integer to each unique tag.
    #   - Text Encoding for text: Depending on the nature of the text, you could use methods like TF-IDF, word embeddings (e.g., Word2Vec, GloVe), or more complex models like BERT to generate numerical representations.
   

    # TODO: figure out how to transform and use it to train on pytorch
    pyg_graph = from_networkx(graphs[0]) 
    print(pyg_graph.x)
    print(pyg_graph.edge_index)
    print(pyg_graph.tag)
    print(pyg_graph.text)

    # TODO: to save dataset
    # torch.save(data, 'graph_data.pt')





def build_graph(xml_root):
    """
    Build a graph from equation in MathML format.

    """
    G = nx.Graph()

    

    def create_node(element,parent_uid=0):
        global uid

        # Adding parent node "math"
        if len(G.nodes) == 0:
            tag = rn(element.tag)
            # G.add_node(0,x=tag,tag=tag,text=element.text)
            G.add_node(0,tag=tag,text=element.text) # set parent node: math with uid=0
            uid = 1                                              # start new nodes from uid=1

        # Go through each child
        for child in element:
            tag = rn(child.tag)
            if tag == "annotation":
                #skip the latex annotation
                continue

            # Add new node and edge between himself and the parent
            G.add_node(uid, x=tag, tag=tag, text=child.text)
            G.add_edge(parent_uid, uid)
            uid += 1

            # Check for children itself and if one or more is found, recursive call
            children = [x for x in child]
            if children:
                create_node(child,uid-1)
    
    create_node(xml_root,0)
    return G


def rn(x):
    """Remove Namespace"""
    return x.replace(r"{http://www.w3.org/1998/Math/MathML}", "")

