import html
import json
import math
import re
import unicodedata
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
    
    # create_embedding_tables('dataset/cleaned_formulas.xml')
    stats.xml_occurences('dataset/cleaned_formulas_katex.xml')
    plot.plot_text_frequency_per_tag("out/text_per_tag_katex.json")
    return

    # Load the MathML equation 
    tree = ET.parse('dataset/equations.xml')
    # tree = ET.parse('dataset/cleaned_formulas.xml')
    root = tree.getroot()

    graphs = []


    # Iterate through each XML equation and create a graph
    for i, formula in enumerate(tqdm(root,desc="Generating Graphs",unit="equations")):

        if debug and i >= 10:
            break
        
        # Build graph and add to list
        G = build_graph(formula)
        if i <= 10:
            plot.plot_graph(G,f"{i}_graph")
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
    pyg_graph = from_networkx(graphs[-1]) 
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
            text = "" if element.text is None else element.text
            # G.add_node(0,x=tag,tag=tag,text=element.text)
            G.add_node(0,tag=tag,text=text) # set parent node: math with uid=0
            uid = 1                                              # start new nodes from uid=1

        # Go through each child
        for child in element:
            tag = rn(child.tag)
            text = "" if child.text is None else child.text
            if tag == "annotation":
                #skip the latex annotation
                continue

            # Add new node and edge between himself and the parent
            G.add_node(uid, tag=tag, text=text)
            G.add_edge(parent_uid, uid)
            uid += 1

            # Check for children itself and if one or more is found, recursive call
            children = [x for x in child]
            if children:
                create_node(child,uid-1)
    
    create_node(xml_root,0)
    return G


def create_embedding_tables(xml_path="dataset/cleaned_formulas.xml", debug=False):

    tree = ET.parse(xml_path)
    root = tree.getroot()
    embedding_table = {tag:set() for tag in MATHML_TAGS}

    bad_things = {"numbers":0}

    def find_in_element(element):    
        
        if "math" in element.tag:
            tag = rn(element.tag)
            text = "" if element.text is None else clean_text(element.text)
            embedding_table[tag].add(text)


        for child in element:
            tag = rn(child.tag)
            text = "" if child.text is None else clean_text(child.text)

            if tag=="mn":
                try:
                    number = float(text)
                except:
                    bad_things["numbers"] +=1

            embedding_table[tag].add(text)
            # if tag=="mtext":
            #     [embedding_table[tag].add(word) for word in text.split()]
            # else:
            #     embedding_table[tag].add(text)
            children = [x for x in child]
            if children:
                find_in_element(child)
    
    # iterate over each XML equation
    for i, formula in enumerate(tqdm(root,desc="Counting occurences",unit="equations")):
        if debug and i>= 10000:
            break

        # Run recursive function
        find_in_element(formula)
    
    for tag,values in embedding_table.items():
        if len(values) > 1:
            print(f"{tag} : {len(values)} - examples : {list(values)[0:5] if len(values)>5 else list(values)}")
    print(bad_things)
    
    # Trasform to dict of lists then save it
    texts_per_tag = {key: list(value) for key,value in embedding_table.items()}
    save.json_dump("out/text_per_tag.json",texts_per_tag)
        

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

