import json
import math
import re
import xml.etree.ElementTree as ET
import networkx as nx
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.data import DataLoader
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
    # tree = ET.parse('out/test.xml')
    tree = ET.parse('dataset/cleaned_formulas.xml')
    root = tree.getroot()


    graphs_dataset = []



    # Iterate through each XML equation and create a graph
    for i, formula in enumerate(tqdm(root,desc="Generating Graphs",unit="equations")):

        if debug and i >= 10:
            break
        
        # Build graph and add to list
        G = build_graph(formula)
        graphs_dataset.append(G)


    # graphs = [build_graph(x) for x in root]


    # nx.write_graphml_xml(G,"out/graph.graphml")
    #TODO: one-hot encode
    #TODO: save the graphs in a format, mumpy array for ex.
    # print(nx.to_numpy_array(G))

    # for i,G in enumerate(graphs):

        # Print graph nodes and edges
        

   

    # TODO: figure out how to transform and use it to train on pytorch
    pyg_graph = from_networkx(graphs_dataset[0])
    print(pyg_graph)
    print(pyg_graph.edge_index)
    print(pyg_graph.tag)
    print(pyg_graph.label)






def build_graph(xml_root):
    """
    Build a graph from equation in MathML format.

    """
    G = nx.Graph()

    

    def create_node(element,parent_uid=0):
        global uid

        # Adding parent node "math"
        if len(G.nodes) == 0:
            G.add_node(0,tag=rn(element.tag),label=element.text) # set parent node: math with uid=0
            uid = 1                                              # start new nodes from uid=1

        # Go through each child
        for child in element:
            tag = rn(child.tag)
            if tag == "annotation":
                #skip the latex annotation
                continue

            # Add new node and edge between himself and the parent
            G.add_node(uid, tag=tag, label=child.text)
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

