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

from config import CONFIG, MATHML_TAGS
from utils import plot
from collections import OrderedDict

from preprocessing.VocabBuilder import VocabBuilder
from utils.plot import plot_loss_graph, plot_training_graphs
from utils.save import json_dump
import pandas as pd
import tempfile

# os.environ["OPENBLAS_NUM_THREADS"] = "128"

def main(model_name="default", sample_latex_set="sample", sample_xml_name="sample", sample_vocab_type="split"):
    dir_path = os.path.join("trained_models",model_name)
    params_path = os.path.join(dir_path,"params.json")
    model_path = os.path.join(dir_path,"checkpoint.pt")
    latex_path = os.path.join("dataset/latex_examples.json")


    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with open(params_path,"r") as f:
        params = json.load(f)
    
    with open(latex_path,"r") as f:
        latex_eqs = json.load(f)

    config = CONFIG
    test_loader = None
    vocab = None  

    if "train_loop_config" in params:
        train_config = params.get("train_loop_config",{})
        params.update(train_config)


    config.update(params)
    max_num_nodes = config.get("max_num_nodes",40)
    xml_name = config.get("xml_name", "debug")
    latex_set = config.get("latex_set","OleehyO")
    vocab_type = config.get("vocab_type","concat")

    # Set to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('CUDA availability:', device)
    
    sample, vocab = load_dataset(sample_xml_name, sample_latex_set, False, False, max_num_nodes, sample_vocab_type, False, False)
    sample_loader = DataLoader(sample, batch_size=256, shuffle=False, num_workers=8)

    metrics, model = test_model(config, model_path, sample_loader, vocab, device)
    sample_recon_z, sample_recon_g = reconstruct(model,sample_loader, device, config)
    print(metrics)

    latent_space_file = os.path.join(dir_path, "latent_space.npy")
    if os.path.exists(latent_space_file):
        test_recon_z = np.load(latent_space_file)
        print(f"Loaded from saved numpy file of size {test_recon_z.shape}")
    else:
        test, vocab = load_dataset(xml_name, latex_set, False, False, max_num_nodes, vocab_type, False, False)
        test_loader = DataLoader(test, batch_size=256, shuffle=False, num_workers=8)

        test_recon_z, test_recon_g = reconstruct(model, test_loader, device, config)
        np.save(latent_space_file, test_recon_z)
        print("Saved latent space to numpy file")

    # select only a few
    np.random.shuffle(test_recon_z)
    test_recon_z = test_recon_z[0:2000]

    sample_indices = [0,1,4,8,9]
    sample_labels = [latex_eqs["train"][i] for i in sample_indices]

    # Prepare data for t-SNE
    all_embeddings = np.concatenate([test_recon_z, sample_recon_z[sample_indices]], axis=0)
    test_labels = np.concatenate([np.zeros(len(test_recon_z)), np.array([1,2,3,4,5])])
    

    tsne_results = apply_tsne(all_embeddings)
    pca_resuts = apply_pca(all_embeddings)

    plot.plot_tsne_n_pca(tsne_results, pca_resuts, test_labels, sample_labels, model_name)



def load_dataset(xml_name, latex_set, debug, force_reload, max_num_nodes, vocab_type, shuffle, split_set = True):
        print("Loading dataset...")
        mathml = MathmlDataset(xml_name,latex_set=latex_set,debug=debug,force_reload=False)
        vocab = VocabBuilder(xml_name,vocab_type=vocab_type, debug=debug, reload_vocab=False)
        dataset = GraphDataset(mathml.xml_dir,vocab, max_num_nodes= max_num_nodes, force_reload=force_reload, debug=debug)

        if split_set:
            _, _, test = dataset.split(shuffle=shuffle)
        else:
            test = dataset

        # test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=8)

        print(f"MathML dataset size: {len(mathml)} equations")
        print(f"Dataset sizes: test= {len(test)} graphs. Max number of nodes per graph= {max_num_nodes}")
        return test, vocab


def test_all_models(model_name="default"):

    dir_path = os.path.join("data","ray_results",model_name)
    out_path = os.path.join("trained_models",model_name)

    experiments_data = []

    

    # generator = torch.Generator().manual_seed(seed_value)

    # load setup config and merge with training params
    config = CONFIG

    test = None
    vocab = None    
    

    # Set to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('CUDA availability:', device)

    for i, folder_name in enumerate(sorted(os.listdir(dir_path))): 
        expe_folder = os.path.join(dir_path, folder_name)

        if os.path.isdir(expe_folder):
            
            # get progress and find best training iteration
            progress_csv_path = os.path.join(expe_folder, 'progress.csv')
            if os.path.exists(progress_csv_path):
                progress_data = pd.read_csv(progress_csv_path)
                best_row = progress_data.loc[progress_data.iloc[:, 5].idxmin()] # val_loss
                best_iteration = best_row.iloc[14] - 1 # iteration

                best_checkpoint_path = os.path.join(expe_folder, f"checkpoint_{str(int(best_iteration)).zfill(6)}", "checkpoint.pt")
                print("Best checkpoint path based on val_loss:", best_checkpoint_path)

            # checkpoints = sorted(os.listdir(expe_folder)) 
            # last_checkpoint_path = os.path.join(expe_folder, checkpoints[-1],"checkpoint.pt")
            # print("Checkpoint: ", last_checkpoint_path)

            json_file = os.path.join(expe_folder, 'params.json')
            if os.path.isfile(json_file):
                with open(json_file, 'r') as f:
                    params = json.load(f)
            else:
                print(f"JSON file not found in {expe_folder}")
                continue

            # print(f"Current configuration: {params}")
            if "train_loop_config" in params:
                config.update(params.get("train_loop_config",{}))
            
            config.update(params)

            xml_name = config.get("xml_name", "debug")
            latex_set = config.get("latex_set","OleehyO")
            vocab_type = config.get("vocab_type","concat")
            debug = config.get("debug",False)
            force_reload = config.get("force_reload", False)
            max_num_nodes = config.get("max_num_nodes",40)
            shuffle = config.get("shuffle",False)

            batch_size = config.get("batch_size",256)

            print("Training #: ", i)

            if test == None:
                test, vocab = load_dataset(xml_name, latex_set, debug, force_reload, max_num_nodes, vocab_type, shuffle)
            

            test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=8)
            metrics, _ = test_model(config, best_checkpoint_path, test_loader, vocab, device)

            combined_data = {**params, **metrics}
            experiments_data.append(combined_data)
    
    df = pd.DataFrame(experiments_data)
    
    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(out_path, 'experiments_results_test_set.csv'), index=False)
    print(df)




def test_model(config:dict, model_path:str, test_loader, vocab, device):
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()

    # Get config for models
    hidden_channels=config.get("hidden_channels",32)
    out_channels= config.get("out_channels",16)
    layers = config.get("num_layers",4)
    layer_type=config.get("layer_type",GCNConv)
    scale_grad_by_freq = config.get("scale_grad_by_freq",True)
    method = config.get("method",None)
    batch_norm = config.get("batch_norm",False)
    sparse_edges = config.get("gen_sparse_edges",True)
    train_edge_features = config.get("train_edge_features",False)
    # latex_set = config.get("latex_set","OleehyO")
    # vocab_type = config.get("vocab_type","concat")
    # shuffle = config.get("shuffle",False)
    # debug = config.get("debug",False)
    # xml_name = config.get("xml_name", "debug")
    # force_reload = config.get("force_reload", True)
    # epochs = config.get("num_epochs",200)
    # max_num_nodes = config.get("max_num_nodes",40)
    # # sample_edges = config.get("sample_edges","sparse")
    # mn_type = config.get("mn_type","embed")

    if isinstance(layer_type, str):
        layer_type = load_class_from_string(pyg_nn, layer_type)

    # Build method dict
    if method is None:
        embed = config.get("embed_method","onehot")

        if embed == "freq_embed":
            embed = "embed"
            scale_grad_by_freq = True
        
        method = {
            "onehot":{},
            "embed":{},
            "linear":{},
            "loss": "cross_entropy" if embed == "onehot" else "cosine",
            # "loss": "cross_entropy" if embed == "onehot" else config.get("loss","cosine"),
            "scale": "log",
        }
        # for component in ["tag","concat","combined","mn","mi","mo","mtext","split","pos",]:
        # for component in ["combined","tag","pos"]:
        for component in ["tag","concat","pos","combined","split"]:
            dim = config.get(f"{component}_dim",None)
            if dim is not None:
                if component == "split":
                    method[embed].update({
                        "mi":dim,
                        "mo":dim,
                        "mtext":dim,
                        "mn":dim
                    })
                    # if mn_type == "linear":
                    #     method["linear"] = {"mn":dim}
                    # else:
                    #     method["embed"] = {"mn":dim}
                else:
                    method[embed].update({component:dim})

    else:
        for embed_method in ["onehot","embed","linear"]:
            od_embed = {}
            # for component in ["tag","mn","mi","mo","mtext","concat","combined","pos"]:
            # for component in ["tag","mi","mo","mtext","mn","concat","combined","pos"]:
            for component in ["combined","tag","pos"]:
                dim = method[embed_method].get(component,None)
                if dim is not None:
                    od_embed[component] = dim

            method[embed_method] = od_embed

    print("The embedding method is: ",method)
    # print("Loading dataset...")

    # load and setup dataset
    # load and setup dataset
       

    # _, _, test = dataset.split(shuffle=shuffle)
    # test_loader = DataLoader(test, batch_size=256, shuffle=False, num_workers=8)

    # print(f"MathML dataset size: {len(mathml)} equations")
    # print(f"Dataset sizes: test= {len(test)} graphs. Max number of nodes per graph= {max_num_nodes}")

    # load models
    embedding_dim = sum(method["onehot"].values()) + sum(method["embed"].values()) + sum(method["linear"].values())
    

    encoder = GraphEncoder(embedding_dim,hidden_channels,out_channels,layers,layer_type,batch_norm)
    decoder = GraphDecoder(embedding_dim,hidden_channels,out_channels,layers,layer_type, edge_dim=1,batch_norm=batch_norm)
    model = GraphVAE(encoder, decoder, vocab.shape(), method, scale_grad_by_freq, sparse_edges, train_edge_features)

    # decoder = Decoder(encoder.embedding.weight.data)

    # Load model and weights
    # model = pyg_nn.VGAE(encoder) if config.get("variational",False) else pyg_nn.GAE(encoder)
    # model_state_data = torch.load(model_path)
    # model.load_state_dict(model_state_data["model_state"])
    # Load model weights
    try:
        model_state_data = torch.load(model_path)
        model.load_state_dict(model_state_data["model_state"])
    except KeyError:
        print(f"Error: 'model_state' not found in checkpoint {model_path}")
        return {}, model
    except RuntimeError as e:
        print(f"Error loading model state dict: {e}")
        return {}, model
    
    model.to(device)

    print("Starting testing...")
    loss, auc, ap, acc, sim = validate(model,test_loader,device,config)

    metrics = {"loss": loss, "auc": auc, "ap": ap, "acc": acc, "sim": sim}
    print(metrics)
    return metrics, model

    # node_acc, edge_acc = reconstruct_graph(model,test_loader,device)
    # print("Node accuracy: ", node_acc)
    # print("Edge accuracy: ", edge_acc)

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

def reconstruct(model:GraphVAE,data_loader,device,config, max_num_batches = 50): 
    model.eval()

    # Getting params
    train_edge_features = config.get("train_edge_features",False)

    latent_z = []
    recon_g = []

    with torch.no_grad():
        for i,batch in enumerate(data_loader):
            if i >= max_num_batches:
                break

            batch = batch.to(device)
            edge_weight = batch.edge_attr.to(device) if train_edge_features else None
            
            # Encode
            x = model.embed_x(batch.x,batch.tag,batch.pos,batch.nums).to(device)          
            z = model.encode(x, batch.edge_index,edge_weight)

            graph_embedding = pyg_nn.global_mean_pool(z, batch.batch)

            # Decode
            x_recon, edge_index_recon, e_recon = model.decode_all(z, batch.edge_index)
            recon_data, raw_data = model.reverse_embed_x(x_recon)

            latent_z.append(graph_embedding.cpu().detach().numpy())
            recon_g.append({
                "x":            x_recon.cpu().detach().numpy(),
                "edge_index":   edge_index_recon.cpu().detach().numpy(),
                "e_recon":      None if e_recon is None else e_recon.cpu().detach().numpy(),
                "x_data":       recon_data,
                "raw_x_data":   raw_data
            })

    return np.concatenate(latent_z, axis=0), recon_g



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
    tsne = TSNE(n_components=2, perplexity=30, n_iter=3000,random_state=42) #, learning_rate='auto', init='pca', perplexity=30)
    return tsne.fit_transform(embeddings)

def apply_pca(embeddings):
    pca = PCA(n_components=2,random_state=42)
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

  