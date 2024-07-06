import html
import json
import math
import os
import unicodedata
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.data import DataLoader, Data
from tqdm import tqdm
from config import MATHML_TAGS




class GraphEmbedder:
    def __init__(self, path="out/text_per_tag_katex.json",embedding_dims=10,scale_by_freq=False):
        self.path = os.path.join("/data/nsam947/Freshwater-Modelling",path)

        

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
         
        self.minimum = 0
        self.maximum = 1e6
        # self.max_bound = self.maximum * 1 # set a maximum bound to accept data over the normalisation
        self.linear_transform = nn.Linear(1,embedding_dims)

    def normalise_number(self,number,min_bound,max_bound):  
        try:
            number = float(number)
        except:
            return None
        if number >= max_bound:
            return None
        return (number - min_bound) / (max_bound - min_bound)
    
    def get_bounded_number(self,number,min_bound=0.,max_bound=1e7):
        try:
            number = float(number)
            if number > max_bound or number < min_bound:
                return None
            else:
                return number
        except:
            return None
    
    def log_transform(self,number,c=1e-7):
        try:
            return math.log10(number + c)
        except:
            return None

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
            # tags_indices[MATHML_TAGS.index("mn")] = - 1 # set to -1 to differenciate from the actual 0 number # TODO: it is inactivated for now because log(x+1e7) will ALMOST never give 0
            tag = node[1]['tag']
            text = node[1]['text']
            if tag in self.text_to_idx: # if "mi","mo","mtext"
                # sets the index of the text found in the table
                tags_indices[MATHML_TAGS.index(tag)] = self.text_to_index(text, tag) 
            elif tag == "mn":
                try:
                    # normed_num = self.normalise_number(text)
                    number = self.get_bounded_number(text)
                    transformed = self.log_transform(number) # TODO: create a transformation function to try them all!!
                    # normed = self.normalise_number(transformed,-7,7)
                    if transformed != None:
                        # sets the transformed value in the vector
                        tags_indices[MATHML_TAGS.index(tag)] = transformed
                    else:
                        # print(f"skipped because of {text}")
                        return None
                except:
                    # print(f"Exception because of {text}")
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
                # negative_indices = torch.nonzero(tag_indices == -1).squeeze() 
                zero_indices = torch.nonzero(tag_indices == 0).squeeze() 
                # print(zero_indices)
                
                # Transform linearly everything
                transformed = self.linear_transform(tag_indices.unsqueeze(1))
                transformed[zero_indices] = torch.zeros(self.embedding_dims)

                # transformed = transformed.detach()  # Detach the tensor
                # transformed.requires_grad_(False)  # Ensure requires_grad is False
                embedded_features.append(transformed)

            else:
                tag_indices = tag_indices.unsqueeze(1)
                # tag_indices.requires_grad_(False)  # Ensure requires_grad is False
                # tag_indices = tag_indices.detach()  # Detach the tensor
                embedded_features.append(tag_indices)
                # print(tag_indices.shape)

        embedded_features = torch.cat(embedded_features,dim=1)

        # mn_indices = indices[:,MATHML_TAGS.index("mn")]
        # _indices = [i for i, x in enumerate(mn_indices) if x != 0.]
        # if len(_indices) != 0:
        #     print(mn_indices[_indices])
        #     print(embedded_features[_indices,19:29])
        # TODO: return a new list of MATHML_TAGS, for which when it's mn,mi,mo,mtext, it's duplicated
        # return torch.cat(embedded_features,dim=1)
        return embedded_features



    
