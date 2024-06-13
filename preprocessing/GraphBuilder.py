import html
import json
import networkx as nx
import unicodedata
import xml.etree.ElementTree as ET
import torch
from torch_geometric.data import Data
import numpy as np


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


class GraphBuilder:
    def __init__(self,vocab_path="out/vocab_texts_katex.json",graph_type="Graph"):
        self.mathml_tags = MATHML_TAGS
        self.graph_type = graph_type if graph_type in ["DiGraph","MultiGraph","MultiDiGraph","Graph"] else "Graph"

        with open(vocab_path,"r") as f:
            self.text_to_idx = json.load(f)
        
        self.inactive_id = 0
        self.unknown_id = 1
        self.vocab_size = len(self.text_to_idx)


    def build_graph(self,xml_root):
        """
        Build a graph from equation in MathML format.

        """

        if self.graph_type == "DiGraph":
            G = nx.DiGraph()
        elif self.graph_type =="MultiGraph":
            G = nx.MultiGraph()
        elif self.graph_type == "MultiDiGraph":
            G = nx.MultiDiGraph()
        else:
            G = nx.Graph()
        

        def create_node(element,parent_uid=0):
            global uid

            # Adding parent node "math"
            if len(G.nodes) == 0:
                tag = self.rn(element.tag)
                text = "" if element.text is None else self.clean_text(element.text)

                # x = embedder.create_embedding(tag,text) # not a good position

                G.add_node(0,tag=tag,text=text,pos=0) # set parent node: math with uid=0
                uid = 1                                              # start new nodes from uid=1

            # Go through each child
            for i, child in enumerate(element):
                # Set tag and text
                tag = self.rn(child.tag)
                text = "" if child.text is None else self.clean_text(child.text)
                pos = i

                # Unwanted Latex tag
                if tag == "annotation": 
                    continue

                # Add new node and edge between himself and the parent
                G.add_node(uid, tag=tag, text=text,pos=pos)
                G.add_edge(parent_uid, uid)
                uid += 1

                # Check for children itself and if one or more is found, recursive call
                children = [x for x in child]
                if children:
                    create_node(child,uid-1)
        
        create_node(xml_root,0)
        return G
    
    def index_texts_in_graph(self, G):
        """
        Creates a new Node param termed "indices", a list of len(MATHML_TAGS) + 1 = 32, 
        that one-hots encode the index of the text corresponding to the tag index in MATHML_TAGS + the value of the text in the last position
        """
        for node in G.nodes(data=True):
            tags_indices = np.zeros(len(MATHML_TAGS) + 1)
            tag = node[1]['tag']
            text = node[1]['text']

            # one hot-encode
            tags_indices[MATHML_TAGS.index(tag)] = 1

            # Set the index at the last index
            if tag in self.text_to_idx: 
                tags_indices[-1] = self.text_to_index(text, tag) 
            node[1]['indices'] = tags_indices
        return G

    def convert_to_pyg(self,G):
        node_features = np.array([node[1]['indices'] for node in G.nodes(data=True)],dtype=np.float32)
        x = torch.tensor(node_features,dtype=torch.float32)

        # Extract position in leaf
        pos_list = np.array([node[1]['pos'] for node in G.nodes(data=True)],dtype=np.float32)
        pos = torch.tensor(pos_list,dtype=torch.long)
        
        # Extract edge index
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()

        # For undirected graphs, make sure to add edges in both directions
        if not G.is_directed():
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            
        # Create the PyG Data object
        pyg_graph = Data(x=x, edge_index=edge_index, pos=pos)

        return pyg_graph
    
    def text_to_index(self, text, tag):
        """
        Finds the index of the text in the table corresponding to the tag
        """
        return self.text_to_idx[tag].get(text, self.unknown_id)

    def decode_xml_entities(self,text):
        return html.unescape(text)

    def normalize_unicode(self, text):
        return unicodedata.normalize('NFKC', text)

    def clean_text(self,text):
        text = self.decode_xml_entities(text)
        # text = normalize_unicode(text) # Maybe not?

        text = text.replace('\u00a0',' ').strip()
        return text

    def rn(self,x):
        """Remove Namespace"""
        return x.replace(r"{http://www.w3.org/1998/Math/MathML}", "")