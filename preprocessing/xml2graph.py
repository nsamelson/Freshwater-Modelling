import html
import json
import math
import re
import unicodedata
import xml.etree.ElementTree as ET
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.data import DataLoader, Data
from tqdm import tqdm
# from utils.plot import plot_labels_frequency
from utils import save, stats, plot
from preprocessing.GraphEmbedder import GraphEmbedder
# import torch_geometric.nn as pyg_nn
# import torch_geometric.utils as pyg_utils 
# import torch_geometric.transforms as T
# import torch
# from tensorboardX import SummaryWriter
# from sklearn.manifold import TSNE



def main(generate_stats=False, debug=True):
    """
    Generate a graph from mathml XML equations. 
    Source : https://github.com/Whadup/arxiv_learning/blob/ecml/arxiv_learning/data/load_mathml.py
    """

    # Load the MathML equation 
    tree = ET.parse('dataset/cleaned_formulas_katex.xml')
    root = tree.getroot()

    pyg_graphs = []

    embedder = GraphEmbedder()

    # Iterate through each XML equation and create a graph
    for i, formula in enumerate(tqdm(root,desc="Generating Graphs",unit="equations")):

        if debug and i >= 500:
            break
        
        # Build graph, embed it and convert to torch
        G = build_graph(formula)
        G = embedder.index_texts_in_graph(G)

        if G == None:
            continue
        pyg_graph = convert_to_pyg(G)

        # Add to graph list
        pyg_graphs.append(pyg_graph)

    print(f"Successfully generated {len(pyg_graphs)} out of {len(root)}")

    # pyg_G = pyg_graphs[0]
    # print("Pytorch Graph", pyg_G)
    # print("TAGS: ", pyg_G.tag)
    # print("TEXTs", pyg_G.text)
    # print("X: ",pyg_G.x)
    # print("EDGES: ",pyg_G.edge_index)

    # Save dataset
    torch.save(pyg_graphs, 'dataset/graph_formulas_katex.pt')
    # data = {}
    # data["edges"] = np.array([graph.edge_index for graph in pyg_graphs])
    # data["x"] = np.array([graph.x for graph in pyg_graphs])
    # np.save("dataset/graph_formulas_katex.npz",data)


def convert_to_pyg(G):
    node_features = [node[1]['indices'] for node in G.nodes(data=True)]
    # x = torch.tensor(node_features, dtype=torch.long)
    x = np.array(node_features,dtype=np.float128)
    pyg_graph = from_networkx(G)
    pyg_graph.x = x
    return pyg_graph

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
            text = "" if element.text is None else clean_text(element.text)

            # x = embedder.create_embedding(tag,text) # not a good position

            G.add_node(0,tag=tag,text=text,) # set parent node: math with uid=0
            uid = 1                                              # start new nodes from uid=1

        # Go through each child
        for child in element:
            # Set tag and text
            tag = rn(child.tag)
            text = "" if child.text is None else clean_text(child.text)

            # Unwanted Latex tag
            if tag == "annotation": 
                continue

            # Add new node and edge between himself and the parent
            G.add_node(uid, tag=tag, text=text,)
            G.add_edge(parent_uid, uid)
            uid += 1

            # Check for children itself and if one or more is found, recursive call
            children = [x for x in child]
            if children:
                create_node(child,uid-1)
    
    create_node(xml_root,0)
    return G


def decode_xml_entities(text):
    return html.unescape(text)

def normalize_unicode(text):
    return unicodedata.normalize('NFKC', text)

def clean_text(text):
    text = decode_xml_entities(text)
    # text = normalize_unicode(text) # Maybe not?

    text = text.replace('\u00a0',' ').strip()
    return text

def rn(x):
    """Remove Namespace"""
    return x.replace(r"{http://www.w3.org/1998/Math/MathML}", "")

