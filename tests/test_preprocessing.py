import json
import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch_geometric.nn as pyg_nn
# import xml.etree.ElementTree as ET
# from models.Graph.GraphAutoEncoder import GraphEncoder
# from models.Graph.GraphDataset import GraphDataset
# # from preprocessing.GraphEmbedder import GraphEmbedder
# from preprocessing.GraphBuilder import GraphBuilder
# # from preprocessing.xml2graph import build_graph, convert_to_pyg, rn
# from utils import plot
# from sklearn.preprocessing import PowerTransformer, RobustScaler
# from torch_geometric.data import Data
# import networkx as nx
import unittest




class Test_TestIncrementDecrement(unittest.TestCase):


    def test_increment(self):
        self.assertEqual(4,4)




if __name__=="__main__":
    unittest.main()




# def log_transform(number):
#     return np.log10(number + 1e-7)
#     # return torch.log(torch.tensor(number + 1e-7))

# def inverse_log_transform(tensor):
#     return torch.exp(tensor) - 1e-7

# def mean_normalize(number, mean, range_val):
#     return (number - mean) / range_val

# def inverse_mean_normalize(normalized_number, mean, range_val):
#     return (normalized_number * range_val) + mean


# def normalize(number,min_val=0,max_val=1e6):
#     return (number - min_val) / (max_val - min_val)


# def main():

#     dataset = GraphDataset(root='dataset/', equations_file='cleaned_formulas_katex.xml',force_reload=False, debug=False,)
#     in_features = dataset.num_features
#     vocab_size = dataset.vocab_size
#     print(in_features)
#     model = pyg_nn.GAE(GraphEncoder(in_features, 32, 16,vocab_size=vocab_size,embedding_dim=10))

#     x= dataset[0].x
#     edges = dataset[0].edge_index
#     print(x.shape,edges.shape)
    
#     z = model.encode(x,edges)
#     loss = model.recon_loss(z,edges)
#     print(z.shape)
#     print(z)
#     print(loss)

#     return
#     # tree = ET.parse('dataset/small.xml')
#     # root = tree.getroot()

#     # builder = GraphBuilder()
#     # G = builder.build_graph(root[1])
#     # # print(G.nodes(data=True))

#     # G = builder.index_texts_in_graph(G)
#     # # print(G.nodes(data=True))

#     # py_g = builder.convert_to_pyg(G)
#     # print(py_g)
#     # # print(py_g)
#     # # print(builder.text_to_idx)
#     # # print(builder.vocab_size)

#     # vocab_size = builder.vocab_size
#     # embedding = nn.Embedding(vocab_size,10,padding_idx=0,scale_grad_by_freq=True)
#     # x = embedding(py_g.x[:,-1])

#     # other = torch.cat((py_g.x[:,:-1],x),dim=1)

#     # print(other.shape)


#     embedder = GraphEmbedder()

#     G = build_graph(root[1])
#     G = embedder.index_texts_in_graph(G)
#     pyg_graph = convert_to_pyg(G)
#     # print(pyg_graph.x)

#     with open("out/xml_texts_katex.json","r") as f:
#         texts = json.load(f)
    
#     most_occurent = [text for text,val in texts.items() if val >=20]
#     alphabet = {text:i for i,text in enumerate(most_occurent) }
#     print(alphabet)


#     nodes = create_node_pfhaler(root[1])
#     num_nodes, e = generate_skeleton(nodes)
#     X = torch.zeros((num_nodes, 1), dtype=torch.int64)
#     pos = torch.zeros((num_nodes, 1), dtype=torch.int64)
#     fill_skeleton(nodes, X, pos, alphabet)
#     edge_features = torch.zeros((len(e), 1), dtype=torch.int64)
#     edges = torch.zeros((2, len(e)), dtype=torch.int64)

#     Gp = nx.Graph()
#     # Gp.nodes = X
#     new_x = [(i,{"x":text}) for i,text in enumerate(X)]
#     Gp.add_nodes_from(new_x)
#     for k, (i, j, y) in enumerate(e):
#         edges[0, k] = i
#         edges[1, k] = j
#         edge_features[k, 0] = y
#         Gp.add_edge(i,j)

#     # Gp.edges = list(edges)
    
#     # print(edge_features)
    
#     graph_pfhaler = Data(x=X, edge_index=edges, edge_attr=edge_features, pos=pos)
#     # print(graph_pfhaler.edge_index)
#     # plot.plot_from_pyg(graph_pfhaler, "1_pyg")

#     print(graph_pfhaler.x)


#     plot.plot_graph(G,f"{1}_graph")
#     plot.plot_graph(Gp,f"{0}_graph")
    # for formula in root:

    # seed_value = 0
    # random.seed(seed_value)
    # np.random.seed(seed_value)
    # torch.manual_seed(seed_value)
    # torch.cuda.manual_seed(seed_value)
    # torch.cuda.manual_seed_all(seed_value)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # with open("out/text_per_tag_katex.json","r") as f:
    #     text_occurences_per_tag = json.load(f)

    # num_values = text_occurences_per_tag["mn"]
    # num_list = []
    # for key, value in num_values.items():
    #     try:
    #         key = float(key)
    #         if key <= 1e6:
    #             num_list += [key] * value
    #     except:
    #         continue

    # # numbers_occ = [[float(key)]*value for key, value in numbers.items() if float(key)!= None]
    # # flat_numbers = [x for xs in numbers_occ for x in xs]
    # num_vec = np.array(num_list,dtype=np.float32)
    # np.random.shuffle(num_vec)
    # numbers = num_vec.reshape(-1,16)

    # # print(numbers[0])

    # dtype = torch.float32
    # # numbers = np.array([[0, 1e-2, 2e-1, 1],
    # #                     [3.14, 5e3, 5e2, 0.01]],dtype=np.float32)

    # # min-max normalisation
    # normed = normalize(numbers)

    # # Mean normalisation
    # mean_val = np.mean(numbers)
    # range_val = np.ptp(numbers)  # Range is max - min
    # mean_normed = mean_normalize(numbers, mean_val, range_val)

    # # Log transformation
    # log_transformed = log_transform(numbers)

    # # robust scaling
    # scaler = RobustScaler()
    # scaled = scaler.fit_transform(numbers)
    # # print(numbers[12],scaled[12])
    # reverse_scaled = scaler.inverse_transform(scaled)

    # # box-cox transfo
    # pt = PowerTransformer(method='yeo-johnson', standardize=False)
    # power_transformed = pt.fit_transform(numbers)
    # reverse_powered = pt.inverse_transform(power_transformed)


    # print("Original numbers:", numbers)
    # print("-----------")
    # print("Min-max normalized:", normed)
    # print("-----------")
    # print("Mean normalized:", mean_normed)
    # print("-----------")
    # print("Log transformed:", log_transformed)
    # print("-----------")
    # print("Robust scaled:", scaled)
    # print("Reverse scaled:", reverse_scaled)
    # print("-----------")
    # print("Power transfo:", power_transformed)
    # print("Reverse powered:", reverse_powered)

    # big_numbers = {
    #     "original":numbers,
    #     "min-max_normalisation":normed,
    #     "mean_normalisation":mean_normed,
    #     "log_transform":log_transformed,
    #     "robust_scaling":scaled,
    #     "power_transformation":power_transformed
    # }

    # plot.plot_multiple_distributions(big_numbers)

# CONTENT_SYMBOLS = 192
# ATTRIBUTE_SYMBOLS = 32
# TAG_SYMBOLS = 32
# VOCAB_SYMBOLS = 512
# DIM = CONTENT_SYMBOLS + ATTRIBUTE_SYMBOLS + TAG_SYMBOLS
# MAX_POS = 256

# def create_node_pfhaler(d):
#     """Convert XML-Structure to Python dict-based representation"""
#     children = [x for x in d]
#     if children:
#         children = []
#         for x in d:
#             tag = rn(x.tag)
#             if tag == "annotation":
#                 #skip the latex annotation
#                 continue
#             children.append(create_node_pfhaler(x))
#         return dict(type=rn(d.tag),
#                     children=children,
#                     content="" if d.text is None else d.text)
#     return dict(type=rn(d.tag),
#                 content="" if d.text is None else d.text,
#                 attributes=["=".join(y)for y in d.attrib.items()])
# def generate_skeleton(tree, edges=None, start=0):
#     """
#     Generates the edges of the given tree (represented in python dicts)
#     Returns the number of nodes in the tree as well as the list of edges
#     """
#     if edges is None:
#         e = []
#         return generate_skeleton(tree, edges=e, start=0), e
#     tree["_id"] = start
#     newstart = start + 1
#     if "children" in tree:
#         # edges.append((start, start, 0))
#         for i, child in enumerate(tree["children"]):
#             edges.append((start, newstart, 1)) # topwdown
#             edges.append((newstart, start, 0)) # bottomup
#             newstart = generate_skeleton(child, edges=edges, start=newstart)
#     return newstart
# def fill_skeleton(tree, X, pos, alphabet):
#     """Convert a tree and fills a torch tensor with features derived from the tree"""
#     vocab = alphabet
#     i = tree["_id"]
    
#     # representation_string, without_attr = build_representation_string(tree)
#     representation_string = tree.get("type","") + "_" + tree.get("content","") + "_" + tree.get("type","")
#     without_attr = tree.get("content","")
#     if representation_string in vocab:
#         X[i, 0] = vocab[representation_string]
#     elif without_attr in vocab:
#         X[i, 0] = vocab[without_attr]
#     else:
#         X[i, 0] = VOCAB_SYMBOLS - 1
#     if "children" in tree:
#         for i, child in enumerate(tree["children"]):
#             pos[child["_id"]] = min(i, MAX_POS - 1)
#             fill_skeleton(child, X, pos, alphabet)

# def build_representation_string(obj):
#     representation = ""
#     if "type" in obj:
#         representation+=obj["type"]+"_"
#     if "content" in obj:
#         for char in obj["content"]:
#             representation += char
#         representation += "_"
#     without_attr = representation
#     if "attributes" in obj:
#         for attrib in sorted(obj["attributes"]):
#             if attrib.endswith("em"):
#                 #parse and round
#                 attrib = attrib.split("=")
#                 attrib[-1] = str(round(float(attrib[-1][:-2]), 1)) + "em"
#                 attrib = "=".join(attrib)
#             representation+=attrib+","
#     return representation, without_attr