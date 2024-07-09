import os
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
import xml.etree.ElementTree as ET
import networkx as nx
from sklearn.model_selection import train_test_split

from config import MATHML_TAGS
from preprocessing.VocabBuilder import VocabBuilder, rn, clean_text

GRAPH_TYPES = [
    "Graph",
    "DiGraph",
]

ALGO_TYPES = [
    "TreeGraph",
    "Operator" #TODO later
]

class GraphDataset(InMemoryDataset):
    def __init__(self, root, vocab:VocabBuilder, graph_type = "Graph", max_num_nodes=50, representation="TreeGraph", transform=None, pre_transform=None, pre_filter=None,log=False, force_reload= False, debug=False):
        self.debug = debug
        self.vocab = vocab      
        self.max_num_nodes = max_num_nodes
        self.graph_type = graph_type if graph_type in GRAPH_TYPES else "Graph"  
        self.representation = representation if representation in ALGO_TYPES else "TreeGraph"
        
        self.inactive_id = 0
        self.unknown_id = 1

        self.xml_path = os.path.join(root, "raw/equations.xml")     

        super(GraphDataset, self).__init__(root, transform, pre_transform,pre_filter,log,force_reload)
        # self.data, self.slices = torch.load(self.processed_paths[0])
        self.data, self.slices, self._graph_list = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return [self.xml_path]
    
    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def get_graph_list(self):
        """Returns the graph list containing networkx Graph objects."""
        return self._graph_list

    def _set_graph_list(self, graph_list):
        self._graph_list = graph_list
    
    def process(self):
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        data_list = []
        graph_list = []

        for i, formula in enumerate(tqdm(root, desc="Generating Graphs", unit=" equations", total=len(root))):
            if self.debug and i>= 10:
                break
            
            G, py_g = self.build_graph(formula)
            if G is None or len(py_g.x) > self.max_num_nodes:
                continue

            data_list.append(py_g)
            graph_list.append(G)
        
        data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])
        self._graph_list = graph_list

        print("Saving data...")
        torch.save((data, slices, graph_list), self.processed_paths[0])

    def split(self, train_ratio=0.8, val_ratio=0.1, shuffle= True):
        """
        Splits the dataset into train, validation, and test sets.
        
        Args:
            train_ratio (float): Ratio of the dataset to be used for training.
            val_ratio (float): Ratio of the dataset to be used for validation.
            test_ratio (float): Ratio of the dataset to be used for testing.
        
        Returns:
            tuple: Three tuples containing train, validation, and test sets respectively.
        """
        dataset_size = len(self)
        indices = list(range(dataset_size))
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)

        train_indices, remaining_indices = train_test_split(indices, train_size=train_size, random_state=42, shuffle=shuffle)
        val_indices, test_indices = train_test_split(remaining_indices, train_size=val_size, random_state=42, shuffle=shuffle)

        # split the dataset
        train_dataset = self[torch.tensor(train_indices)]
        val_dataset = self[torch.tensor(val_indices)]
        test_dataset = self[torch.tensor(test_indices)]

        # split corresponding graphs into 3 sets as well
        train_graph_list = [self._graph_list[idx] for idx in train_indices]
        val_graph_list = [self._graph_list[idx] for idx in val_indices]
        test_graph_list = [self._graph_list[idx] for idx in test_indices]

        # update each dataset's graph list
        train_dataset._set_graph_list(train_graph_list)
        val_dataset._set_graph_list(val_graph_list)
        test_dataset._set_graph_list(test_graph_list)

        return train_dataset, val_dataset, test_dataset

    def build_graph(self, xml_root):
        """
        Build a networkx graph and a corresponding PyTorch Geometric Data object from XML data.

        Args:
            xml_root (Element): The root element of the XML structure containing mathematical formula data.

        Returns:
            tuple: A tuple containing a networkx Graph (G) and a PyTorch Geometric Data object (py_g).
        """
        if self.graph_type == "Graph":
            G = nx.Graph()
        elif self.graph_type =="DiGraph":
            G = nx.DiGraph()
        else:
            G = nx.Graph()

        x, positions, tags, nums = [], [], [], []

        def add_to_list(index, tag_id, pos, num):
            """Helper function to add node attributes to lists."""
            x.append(index)
            tags.append(tag_id)
            positions.append(pos)
            nums.append(num)
        
        def create_node(element,parent_uid=0):
            """Recursive function to create nodes and edges in the graph."""
            global uid

            # Adding parent node "math"
            if len(G.nodes) == 0:
                tag = rn(element.tag)
                text = "" if element.text is None else clean_text(element.text)
                index, tag_index = self.get_index_from_vocab(tag,text)

                G.add_node(0,tag=tag,text=text,pos=0, index=index, num=-1) # set parent node: math with uid=0
                add_to_list(index,tag_index,pos=0,num=-1)
                uid = 1                               # start new nodes from uid=1

            # Go through each child
            for i, child in enumerate(element):
                tag = rn(child.tag)
                text = "" if child.text is None else clean_text(child.text)
                index, tag_index = self.get_index_from_vocab(tag,text)
                pos = max(i,255)
                
                num = -1
                if tag == "mn":
                    try: 
                        num = float(text)
                    except:
                        return None
                    

                # Add new node and edge between himself and the parent
                G.add_node(uid, tag=tag, text=text,pos=pos, index=index, num=num)
                add_to_list(index,tag_index,pos=pos,num=num)
                G.add_edge(parent_uid, uid)
                uid += 1

                # Check for children itself and if one or more is found, recursive call
                children = [x for x in child]
                if children:
                    create_node(child,uid-1)
        
        create_node(xml_root,0)

        # Extract edge index
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        edge_features = torch.ones((edge_index.shape[1],)) #(edge_index.shape[1],1)
        if not G.is_directed():
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_features = torch.cat([edge_features, torch.zeros(edge_features.shape)],dim=0)
        
        # Create pytorch geometric graph
        py_g = Data(
            x=torch.tensor(x,dtype=torch.long),
            edge_index= edge_index,
            edge_attr= edge_features,
            tag_index = torch.tensor(tags,dtype=torch.long),
            pos=torch.tensor(positions,dtype=torch.long),
            nums = torch.tensor(nums,dtype=torch.float32),
        )

        return G, py_g
    
    def get_index_from_vocab(self, tag, text):
        """
        Return an index based on the type of vocabulary used.

        Args:
            tag (str): The tag associated with the element.
            text (str): The text content of the element.

        Returns:
            tuple: A tuple containing two indices:
                - index: Index based on the vocabulary type. Returns 0 if the element is empty,
                            returns 1 if the element is not found in the vocabulary.
                - tag_index: Index of the tag in the predefined MATHML_TAGS list.
        """
        index = 0
        tag_index = MATHML_TAGS.index(tag)

        if self.vocab.vocab_type == "combined":
            index = self.vocab.vocab_table.get(text,self.unknown_id)
        elif self.vocab.vocab_type == "concat":
            if text == "":
                index = self.vocab.vocab_table.get(tag,self.unknown_id)
            else:
                concat = "_".join([tag,text])
                index = self.vocab.vocab_table.get(concat,self.unknown_id)
        elif self.vocab.vocab_type == "split":
            if tag in self.vocab.vocab_table.keys():
                index = self.vocab.vocab_table[tag].get(text,self.unknown_id)
        
        return index, tag_index
    
    

