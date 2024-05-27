import html
import json
import math
import unicodedata
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.data import DataLoader, Data
from tqdm import tqdm

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



class GraphEmbedder:
    def __init__(self, path="out/text_per_tag_katex.json",embedding_dims=10,scale_by_freq=False):
        self.path = path

        with open(self.path,"r") as f:
            self.text_occurences_per_tag = json.load(f)
        
        # Embedding layers and text to index mappings for each tag type
        self.embedding_dims = embedding_dims
        self.embeddings = {}
        self.text_to_idx = {}

        self.inactive_id = 0
        self.unknown_id = 1

        for tag in ["mi", "mo", "mtext"]:
            texts = self.text_occurences_per_tag[tag]
            texts = [None,"<unk>"] + list(texts.keys())
            self.text_to_idx[tag] = {text: idx for idx, text in enumerate(texts)}
            self.embeddings[tag] = nn.Embedding(len(texts), embedding_dims,scale_grad_by_freq=scale_by_freq)
         
        #TODO: tag "mn" needs to be NORMALISED AND CONVERTED to nn.Linear (probably inside the model)
        self.minimum = 0
        self.maximum = 1e6
        self.max_bound = self.maximum * 10 # set a maximum bound to accept data over the normalisation
        # self.find_extremum_values(self.text_occurences_per_tag["mn"])

    def normalise_number(self,number):  
        try:
            number = float(number)
        except:
            return None
        if number >= self.max_bound:
            return None
        return (number - self.minimum) / (self.maximum - self.minimum)


    def text_to_index(self, text, tag):
        """
        Finds the index of the text in the table corresponding to the tag
        """
        return self.text_to_idx[tag].get(text, self.unknown_id)
    
    def index_texts_in_graph(self, G):
        """
        Creates a new Node param termed "indices", a list of len(MATHML_TAGS) = 31, 
        that one-hots encode the index of the text corresponding to the tag index in MATHML_TAGS
        """
        for node in G.nodes(data=True):
            tags_indices = np.zeros(len(MATHML_TAGS))
            tag = node[1]['tag']
            text = node[1]['text']
            if tag in self.text_to_idx:
                tags_indices[MATHML_TAGS.index(tag)] = self.text_to_index(text, tag)
            elif tag == "mn":
                try:
                    normed_num = self.normalise_number(text)
                    if normed_num != None:
                        tags_indices[MATHML_TAGS.index(tag)] = normed_num
                    else:
                        return None
                except:
                    return None
            else:
                tags_indices[MATHML_TAGS.index(tag)] = 1
            node[1]['indices'] = tags_indices
        return G
    

    def embed_indices(self, indices):
        """
        This is a function to take into account the expansion of the features of certain tags when using nn.Embedding
        TODO: verify this is working correclty, when called in the model
        """
        embedded_features = []
        for i, tag in enumerate(MATHML_TAGS):
            if tag in self.embeddings:
                idx = indices[i]
                if idx == self.inactive_id:
                    embedded_features.append(torch.zeros(self.embedding_dims))
                else:
                    embedded_features.append(self.embeddings[tag](torch.tensor([idx], dtype=torch.long)).squeeze(0))
            else:
                embedded_features.append(torch.tensor([indices[i]]))
        return torch.cat(embedded_features)



    
