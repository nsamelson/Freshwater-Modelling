import json
import math
import re
import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.data import DataLoader
from tqdm import tqdm
from stats.plot_labels import plot_labels_frequency
from utils import save
# import torch_geometric.nn as pyg_nn
# import torch_geometric.utils as pyg_utils 
# import torch_geometric.transforms as T
# import torch
# from tensorboardX import SummaryWriter
# from sklearn.manifold import TSNE



def main(generate_stats=True):
    """
    Generate a graph from mathml XML equations. 
    Source : https://github.com/Whadup/arxiv_learning/blob/ecml/arxiv_learning/data/load_mathml.py
    """
    
    # Load the MathML equation 
    # tree = ET.parse('out/test.xml')
    tree = ET.parse('dataset/equations.xml')
    root = tree.getroot()

    xml_tags = {}
    xml_labels = {}


    # Iterate through each XML equation and create a graph
    for i, formula in enumerate(tqdm(root,desc="Generating Graphs",unit="equations")):
        G = build_graph(formula)

        if generate_stats:
            # get the tags out of the graph and add to the dict
            for _, tag in G.nodes(data="tag"):
                xml_tags[tag] = xml_tags.get(tag, 0) + 1            
            

            # get the labels out of the graph and add to the dict
            for _, label in G.nodes(data="label"):
                xml_labels[label] = xml_labels.get(label, 0) + 1    

    if generate_stats:    
        print("Number of different tags: ", len(xml_tags.keys()))
        print("Number of different labels: ",len(xml_labels.keys()))

        # save into json
        save.json_dump("out/xml_tags_count.json",xml_tags)
        save.json_dump("out/xml_labels_count.json",xml_labels)

        plot_labels_frequency()




    # graphs = [build_graph(x) for x in root]


    # nx.write_graphml_xml(G,"out/graph.graphml")
    #TODO: one-hot encode
    #TODO: save the graphs in a format, mumpy array for ex.
    # print(nx.to_numpy_array(G))

    # for i,G in enumerate(graphs):

        # Print graph nodes and edges
        

    # Draw the graph using NetworkX's built-in drawing functions
    # pos = nx.spring_layout(G)  # Compute graph layout

    # labels = {n: lab["label"] if lab["label"] else lab["tag"] for n,lab in G.nodes(data=True)}

    # nx.draw_networkx_nodes(G, pos=pos, node_size=500, node_color='lightblue')
    # nx.draw_networkx_edges(G, pos=pos, width=1.0, alpha=0.5)
    # nx.draw_networkx_labels(G, pos=pos, labels=labels,font_size=10, font_family='sans-serif')

    # plt.axis('off')

    # # Show the graph
    # plt.title('MathML Structure Graph')
    # plt.savefig(f'out/{i}_graph.jpg', format='jpeg', dpi=300) 

    # # TODO: figure out how to transform and use it to train on pytorch
    # pyg_graph = from_networkx(G)
    # print(pyg_graph)
    # print(pyg_graph.edge_index)
    # print(pyg_graph.tag)
    # print(pyg_graph.label)




def build_graph(xml_root):
    """
    Build a graph from equation in MathML format.

    """
    G = nx.Graph()

    

    def create_node(element,parent_uid=0):
        global uid

        # Adding parent node "Math"
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

