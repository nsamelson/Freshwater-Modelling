import math
import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx, from_networkx
import torch


def main():
    """
    Generate a graph from mathml XML equations. 
    Source : https://github.com/Whadup/arxiv_learning/blob/ecml/arxiv_learning/data/load_mathml.py
    """
    # Parse the MathML equation and account for namespaces
    tree = ET.parse('out/test.xml')
    root = tree.getroot()

    
    base =  {'type': 'math', 'children': [                                  #0
        {'type': 'mrow', 'children': [                                      #1
            {'type': 'msup', 'children': [                                  #2
                {'type': 'mi', 'content': 'A', 'attributes': []},           #3
                {'type': 'mn', 'content': '2', 'attributes': []}            #4
            ], 'content': ''}, 
            {'type': 'mo', 'content': '+', 'attributes': []}                #5
        ], 'content': ''}, 
        {'type': 'mrow', 'children': [                                      #6
            {'type': 'msup', 'children': [                                  #7
                {'type': 'mi', 'content': 'B', 'attributes': []},           #8
                {'type': 'mn', 'content': '2', 'attributes': []}            #9
            ], 'content': ''}, 
            {'type': 'mo', 'content': '=', 'attributes': []}                #10
        ], 'content': ''}, 
        {'type': 'mrow', 'children': [                                      #11
            {'type': 'msup', 'children': [                                  #12
                {'type': 'mi', 'content': 'C', 'attributes': []},           #13
                {'type': 'mn', 'content': '2', 'attributes': []}            #14
            ], 'content': ''}
        ], 'content': ''}
    ], 'content': ''}

    G = nx.MultiDiGraph()
    global uid
    uid = 0
    

    def create_node(element,parent_uid=0):
        global uid

        # Adding parent node "Math"
        if len(G.nodes) == 0:
            G.add_node(uid,tag=rn(element.tag),label=element.text)
            uid += 1

        for child in element:
            tag = rn(child.tag)
            if tag == "annotation":
                #skip the latex annotation
                continue

            G.add_node(uid, tag=tag, label=child.text)
            G.add_edge(parent_uid, uid)
            uid += 1

            children = [x for x in child]
            if children:
                create_node(child,uid-1)



    # Create graph nodes and edges
    create_node(root)
    # nx.write_graphml_xml(G,"out/graph.graphml")
    #TODO: one-hot encode
    #TODO: save the graphs in a format, mumpy array for ex.
    # print(nx.to_numpy_array(G))

    # Print graph nodes and edges
    print("Nodes:", G.nodes(data=True))
    print("Edges:", G.edges())

    # Draw the graph using NetworkX's built-in drawing functions
    pos = nx.spring_layout(G,k = 1/math.sqrt(len(G.nodes.values())))  # Compute graph layout

    labels = {n: lab["tag"]+"_"+lab["label"] if lab["label"] else lab["tag"] for n,lab in G.nodes(data=True)}

    nx.draw_networkx_nodes(G, pos=pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_edges(G, pos=pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos=pos, labels=labels,font_size=10, font_family='sans-serif')

    plt.axis('off')

    # Show the graph
    plt.title('MathML Structure Graph')
    plt.savefig('out/graph.jpg', format='jpeg', dpi=300) 

    # TODO: figure out how to transform and use it to train on pytorch
    pyg_graph = from_networkx(G)
    print(pyg_graph)
    print(pyg_graph.edge_index)
    print(pyg_graph.tag)
    print(pyg_graph.label)





def rn(x):
    """Remove Namespace"""
    return x.replace(r"{http://www.w3.org/1998/Math/MathML}", "")