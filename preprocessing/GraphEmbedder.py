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
        self.linear_transform = nn.Linear(1,10)

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
            tags_indices[MATHML_TAGS.index("mn")] = - 1 # set to -1 to differenciate from the actual 0 number
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
    

    def embed_into_vector(self, indices):
        """
        This is a function to take into account the expansion of the features of certain tags when using nn.Embedding
        """
        embedded_features = []
        for i, tag in enumerate(MATHML_TAGS):
            tag_indices = indices[:, MATHML_TAGS.index(tag)] # List of indices for the specific tag
            
            # mi, mo and mtext
            if tag in self.embeddings:
                tag_indices = tag_indices.long()
                # tag_indices = torch.tensor(tag_indices,dtype=torch.long)
                # find indices that are inactive
                zero_indices = torch.nonzero(tag_indices == 0).squeeze() 
                
                # Embed tags, and then replace the inactive ones with a vector of zeros
                embedded_tags = self.embeddings[tag](tag_indices)
                embedded_tags[zero_indices] = torch.zeros(self.embedding_dims)
                # embedded_tags.requires_grad_(False) # Ensure requires_grad is False
                # embedded_tags = embedded_tags.detach()  # Detach the tensor

                embedded_features.append(embedded_tags)
            
            # mn: numerical values
            elif tag == "mn":

                # find indices that are inactive
                negative_indices = torch.nonzero(tag_indices == -1).squeeze() 
                
                # Transform linearly everything
                transformed = self.linear_transform(tag_indices.unsqueeze(1))
                transformed[negative_indices] = torch.zeros(self.embedding_dims)

                # transformed = transformed.detach()  # Detach the tensor
                # transformed.requires_grad_(False)  # Ensure requires_grad is False
                embedded_features.append(transformed)

            else:
                tag_indices = tag_indices.unsqueeze(1)
                # tag_indices.requires_grad_(False)  # Ensure requires_grad is False
                # tag_indices = tag_indices.detach()  # Detach the tensor
                embedded_features.append(tag_indices)
                # print(tag_indices.shape)

        # TODO: return a new list of MATHML_TAGS, for which when it's mn,mi,mo,mtext, it's duplicated
        return torch.cat(embedded_features,dim=1)



    
