import datetime
import json
import os
from matplotlib import pyplot as plt
import numpy as np
import plotly.express as px
# from sklearn.base import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.manifold import TSNE

from torch_geometric.utils.convert import to_networkx, from_networkx 
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE, GCNConv, GraphConv
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, InMemoryDataset, collate, Batch, Dataset
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import DataLoader
import torch_geometric.nn as pyg_nn
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import tostring
import torch_geometric.transforms as T
from tqdm import tqdm
from models.train import validate
# from preprocessing.GraphEmbedder import GraphEmbedder, MATHML_TAGS
from preprocessing.MathmlDataset import MathmlDataset
from preprocessing.GraphDataset import GraphDataset
from models.Graph.GraphAutoEncoder import GraphEncoder, GraphDecoder, GraphVAE
import random

from config import MATHML_TAGS

from preprocessing.VocabBuilder import VocabBuilder
from utils.plot import plot_loss_graph, plot_training_graphs
from utils.save import json_dump
import pandas as pd
import tempfile

# os.environ["OPENBLAS_NUM_THREADS"] = "128"

def main(model_name="default"):
    dir_path = os.path.join("trained_models",model_name)
    params_path = os.path.join(dir_path,"params.json")
    model_path = os.path.join(dir_path,"checkpoint.pt")

    with open(params_path,"r") as f:
        config = json.load(f)

    test_model(config,model_path)


def test_model(config:dict, model_path:str):
    seed_value = 0
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set to device
    # device = 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('CUDA availability:', device)

    # load and setup dataset
    # root_dir = "/data/nsam947/Freshwater-Modelling/dataset/"
    # dataset = GraphDataset(root=root_dir, equations_file='cleaned_formulas_katex.xml',force_reload=False)  

    # data_size = len(dataset)
    # test_data = dataset[int(data_size * 0.9):]
    # test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8)

    # Get channels for layers
    embedding_dim = config.get("embedding_dims",200)
    # in_channels = dataset.num_features + embedding_dim - 1
    hidden_channels=config.get("hidden_channels",32)
    out_channels= config.get("out_channels",16)
    layers = config.get("num_layers",4)
    scale_grad_by_freq = config.get("scale_grad_by_freq",True)
    layer_type=config.get("layer_type",GCNConv)
    layer_type = load_class_from_string(pyg_nn,layer_type)
    latex_set = config.get("latex_set","OleehyO")
    vocab_type = config.get("vocab_type","concat")
    method = config.get("method",{"onehot":["concat"]})
    xml_name = "default"
    debug = False
    force_reload = False

    # load and setup dataset
    mathml = MathmlDataset(xml_name,latex_set=latex_set,debug=debug, force_reload=False)
    vocab = VocabBuilder(xml_name,vocab_type=vocab_type, debug=debug, reload_vocab=force_reload, reload_xml_elements=False)
    dataset = GraphDataset(mathml.xml_dir,vocab, force_reload=force_reload, debug=debug)

    train, val, test = dataset.split(shuffle=False)
    test_loader = DataLoader(test, batch_size=256, shuffle=False, num_workers=8)

    # load model
    encoder = GraphEncoder(embedding_dim,hidden_channels,out_channels,layers,layer_type)
    decoder = GraphDecoder(embedding_dim,hidden_channels,out_channels,layers,layer_type)
    model = GraphVAE(encoder, decoder, vocab.shape(), embedding_dim,method ,scale_grad_by_freq)

    # decoder = Decoder(encoder.embedding.weight.data)

    # Load model and weights
    # model = pyg_nn.VGAE(encoder) if config.get("variational",False) else pyg_nn.GAE(encoder)
    model_state_data = torch.load(model_path)
    model.load_state_dict(model_state_data["model_state"])
    model.to(device)


    node_acc, edge_acc = reconstruct_graph(model,test_loader,device)
    print("Node accuracy: ", node_acc)
    print("Edge accuracy: ", edge_acc)

    # Evaluate on the test set
    # avg_val_loss, avg_auc, avg_ap = validate(
    #         model,test_loader,device,
    #         variational=config.get("variational",False),
    #         neg_sampling_method=config.get("sample_edges","sparse"),
    #         force_undirected=config.get("force_undirected",True)
    #     )
    # metrics = {"test_loss": avg_val_loss, "auc":avg_auc,"ap":avg_ap}
    # print(f"Model performance on the test set: {metrics}")

    # print(test_loader.shape)
    

   

    # Visualise things
    # visualise_bottleneck(model,test_data, device)
    # new_nodes, edges, old_nodes = reconstruct_graph(model,test_loader,device)
    # new_graph = build_recon_graph(new_nodes, edges, vocab)
    # old_graph = build_recon_graph(old_nodes, edges, vocab)

    # # original equation
    # xml_root = graph_to_xml(old_graph)
    # tree = ET.ElementTree(xml_root)
    # tree.write("out/encoded_equation.xml", encoding="utf-8", xml_declaration=True)

    # # decoded equation
    # xml_root = graph_to_xml(new_graph)
    # tree = ET.ElementTree(xml_root)
    # tree.write("out/decoded_equation.xml", encoding="utf-8", xml_declaration=True)

def generate_all_possible_edges(num_nodes):
    """Generate all possible edges for a graph with num_nodes nodes."""
    row = torch.arange(num_nodes).repeat_interleave(num_nodes)
    col = torch.arange(num_nodes).repeat(num_nodes)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index

def graph_to_xml(G):
    def create_xml_element(node_id, parent_element, visited):
        if node_id in visited:
            return
        visited.add(node_id)

        node_data = G.nodes[node_id]
        tag = node_data['tag']
        text = node_data.get('text', '')

        # Create XML element
        element = ET.SubElement(parent_element, tag)
        if text:
            element.text = text

        # # Attach to parent element
        # parent_element.append(element)

        # Recursively create child elements
        # children = [n for n in G[node_id]]
        # print(children)
        children = [n for n in G[node_id] if n not in visited]
        for child_id in children:
            create_xml_element(child_id, element, visited)

    # Assuming the root node is node 0
    root_data = G.nodes[0]
    root = ET.Element(root_data['tag'], xmlns="http://www.w3.org/1998/Math/MathML")
    if root_data.get('text'):
        root.text = root_data['text']

    visited = set()

    # Recursively build the XML tree
    for child_id in G[0]:
        create_xml_element(child_id, root, visited)

    return root



def build_recon_graph(nodes, edges, vocab):
    new_graph = nx.Graph()

    # Get vocab
    # vocab_path = os.path.join("/data/nsam947/Freshwater-Modelling","out/vocab_texts_katex.json")
    # with open(vocab_path,"r") as f:
    #     vocab = json.load(f)

    vocab_texts = list(vocab.vocab_table.keys())
    
    # Add nodes with features
    for i, features in enumerate(nodes):
        tag_text = vocab_texts[features]

        tag, text = "", ""
        if "_" in tag_text:
            tag, text = tag_text.split("_")
        else:
            tag = tag_text

        # one_hot = features[:-1]
        # mathml_index = np.flatnonzero(one_hot)[0]
        # tag = MATHML_TAGS[mathml_index]

        # text = ""
        # if tag in ["mi","mo","mtext","mn"]:
        #     vocab_index = features[-1]
        #     text = vocab_texts[vocab_index]
        

        new_graph.add_node(i, tag=tag, text=text)

    new_graph.add_edges_from(edges)

    # print(new_graph.nodes(data=True))

    return new_graph

def reconstruct_graph(model:GraphVAE, test_loader, device):
    model.eval()

    equation_index = 420
    tot_node_accuracy = 0
    tot_edge_accuracy = 0

    with torch.no_grad():
        for i,batch in enumerate(test_loader):
            # if i == equation_index:                
            batch = batch.to(device)
            edge_index = batch.edge_index.to(device)

            # Encodeing part
            x = model.embed_x(batch.x,batch.tag_index,batch.pos,batch.nums).to(device)         
            z = model.encode(x, batch.edge_index, batch.edge_attr)

            accuracy = model.calculate_accuracy(z,x,batch.edge_index,batch.edge_attr)

            tot_node_accuracy += accuracy[0]
            tot_edge_accuracy += accuracy[1]
                # print("accuracy: ", accuracy)

                # # Decoding part
                # # edge_index = generate_all_possible_edges(graph.num_nodes).to(device)
                # x_recon, e_recon, ef_recon = model.decode_all(z,edge_index,sigmoid=True)

                # # print("Input graph nodes: ",graph.x, graph.x.shape)
                # # print("Input graph edges: ",edge_index, edge_index.shape)
                # # print("Latent space ",z, z.shape)
                # # print("Output graph nodes: ",x_recon, x_recon.shape)
                # # print("Output graph edges: ", e_recon, e_recon.shape, torch.count_nonzero(e_recon))


                # x_recon = x_recon.cpu().numpy()
                # e_recon = e_recon.cpu().numpy()
                

                # filtered_edge_index = edge_index[:, e_recon == 1]

                # edge_tuples = [(int(filtered_edge_index[0, i]), int(filtered_edge_index[1, i])) for i in range(filtered_edge_index.shape[1])]



                # return x_recon, edge_tuples, batch.x.cpu().numpy()   

        node_accuracy = tot_node_accuracy / len(test_loader)
        edge_accuracy = tot_edge_accuracy / len(test_loader)

        return node_accuracy, edge_accuracy

            # print("New py_graph: ", new_pyg)
            # print("New graph: ", new_graph)
            # print(new_graph.edges())
            # print(new_graph.nodes(data=True))


            # print(graph.x[:,-1])
            # print(x_recon[:,-1])

        # # Use the decoder to reconstruct edge probabilities
        # reconstructed_edges = model.decoder(z, graph.edge_index)
        # print(reconstructed_edges, reconstructed_edges.shape)

        # # Optionally, apply a sigmoid activation to get edge probabilities
        # reconstructed_probs = torch.sigmoid(reconstructed_edges)
        # print(reconstructed_probs, reconstructed_probs.shape)

        # Threshold probabilities to obtain edge predictions
        # edge_predictions = (reconstructed_probs > threshold).float()

        # if model.variational:
        #     z = model.reparameterize(z_mean, z_logstd)

        # edge_probs = model.decode(z, graph.edge_index)
        # print("Reconstructed edge probabilities", edge_probs, edge_probs.shape)
        
        # decoded_indices = model.encoder.find_nearest_embeddings(z)
        # print("Decoded indices", decoded_indices, decoded_indices.shape)

        # combined_features = torch.cat((graph.x[:, :-1], decoded_indices.unsqueeze(1).float()), dim=1)
        # print("Combined features", combined_features, combined_features.shape)




def visualise_bottleneck(model, test_set, device, save_dir="out/", num_clusters=2):
    embeddings, graph_indices = extract_embeddings(model, test_set, device)
    print(f"Loaded embeddings: {embeddings.shape}")

    # Apply t-SNE
    tsne_results = apply_tsne(embeddings)
    print(tsne_results.shape)
    plot_embeddings(tsne_results, graph_indices, "t-SNE Embeddings", os.path.join(save_dir, "tsne_embeddings.png"))

    # Apply PCA
    pca_results = apply_pca(embeddings)
    print(pca_results.shape)
    plot_embeddings(pca_results, graph_indices, "PCA Embeddings", os.path.join(save_dir, "pca_embeddings.png"))



def extract_embeddings(model, dataset, device):
    model.eval()
    all_embeddings = []
    graph_indices = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataset), unit="batch", desc="Encoding batches"):
            if i > 9:
                break  # Stop after the first 20 batches
            data = data.to(device)
            embeddings, _ = model.encode(data.x, data.edge_index)
            all_embeddings.extend(embeddings.detach().cpu().numpy())  # directly convert to numpy and extend the list
            graph_indices.extend([i] * embeddings.shape[0])  # keep track of graph index for each node
            del embeddings  # explicitly delete to free GPU memory
    return np.array(all_embeddings), np.array(graph_indices, dtype=int)



def plot_embeddings(embeddings, labels, title="Embeddings Visualization", save_dir="out/embeddings_visualisation.png"):
    fig = px.scatter(x=embeddings[:, 0], y=embeddings[:, 1], color=labels)
    fig.update_layout(
        title=title,
        xaxis_title="First Component",
        yaxis_title="Second Component",
    )
    fig.write_image(save_dir)

def apply_tsne(embeddings):
    tsne = TSNE(n_components=2, random_state=42) #, learning_rate='auto', init='pca', perplexity=30)
    return tsne.fit_transform(embeddings)

def apply_pca(embeddings):
    pca = PCA(n_components=2)
    return pca.fit_transform(embeddings)

def apply_tsne_with_pca(embeddings, pca_components=50):
    pca = PCA(n_components=pca_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2, random_state=42, learning_rate='auto', init='random', perplexity=30)
    return tsne.fit_transform(reduced_embeddings)


def apply_kmeans(embeddings, num_clusters=2):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels

def count_total_nodes(dataloader):
    total_nodes = 0
    for data in dataloader.dataset:
        total_nodes += data.num_nodes
    return total_nodes



def load_class_from_string(module, class_name):
    return getattr(module, class_name)

# def test_model(model,test_loader,device):
#     color_list = ["red", "orange", "green", "blue", "purple", "brown"]
#     colors = []
#     embs = []


#     model.eval()
#     with torch.no_grad():
#         for batch in test_loader:
#             batch = batch.to(device)


#             # Generate negative edges
#             pos_edge_index = batch.edge_index
#             neg_edge_index = negative_sampling(
#                 edge_index=pos_edge_index, 
#                 num_nodes=batch.num_nodes, 
#                 num_neg_samples=pos_edge_index.size(1)
#             ).to(device)
            
#             z = model.encode(batch.x, pos_edge_index)
#             # pred = model.decode()
#             # colors += [color_list[y] for y in labels]

#     xs, ys = zip(*TSNE().fit_transform(z.cpu().detach().numpy()))
#     plt.scatter(xs, ys, color=colors)
#     plt.show()

  