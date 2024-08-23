import datetime
import json
import os
from matplotlib import pyplot as plt
import numpy as np
# from sklearn.base import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.manifold import TSNE
from ray import tune, train
import ray
from ray.train import Checkpoint, RunConfig #, ScalingConfig
from ray.air import session, config, ScalingConfig
from ray.tune.stopper import TrialPlateauStopper
from ray.train.torch import TorchTrainer


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
import torch_geometric.transforms as T
from tqdm import tqdm
from config import CONFIG
# from preprocessing.GraphEmbedder import GraphEmbedder
from preprocessing.MathmlDataset import MathmlDataset
from preprocessing.GraphDataset import GraphDataset
from preprocessing.VocabBuilder import VocabBuilder
from models.Graph.GraphAutoEncoder import GraphEncoder, GraphVAE, GraphDecoder
import random
# from tensorboardX import SummaryWriter

from utils.plot import plot_loss_graph, plot_training_graphs, plot_training_history
from utils.save import json_dump
import pandas as pd
import tempfile





def main(model_name="GraphVAE_new",latex_set = "OleehyO",vocab_type="concat", xml_name= "default", method= {"onehot":{"concat":256}}, epochs=200, force_reload=False):
    storage_path= "/data/nsam947/Freshwater-Modelling/data/ray_results"
    work_dir = "/data/nsam947/Freshwater-Modelling"
    tmp_dir = "/data/nsam947/tmp"


    os.makedirs(tmp_dir, exist_ok=True)
    os.chmod(tmp_dir, 0o777)  # Adjust permissions as needed
    os.environ["RAY_TMPDIR"] = tmp_dir
    ray.init()


    # trials_dir = os.path.join(storage_path, model_name)
    # checkpoint_dir = os.path.join(trials_dir, "checkpoints")
    print("Current Working Directory:", os.getcwd())



    config = {
        "model_name": model_name,
        "num_epochs": epochs,
        "variational": True,
        "alpha": 1,
        "beta": 1,
        "gamma":1,
        "latex_set":latex_set,
        "vocab_type":vocab_type,
        "method":method,
        "xml_name": xml_name,
        "force_reload": force_reload,
        "train_edge_features":True,
        "num_layers":2,
        "out_channels":32,
        "hidden_channels":512,
    }


    grace_period = 10 if epochs != 1 else 1

    # ray.init()
    scaling_config = ScalingConfig(num_workers=1, use_gpu=True,trainer_resources={"CPU": 8}) # trainer_resources={"cpu":8}
    stopper = TrialPlateauStopper(
        metric="val_loss",
        std=0.005,
        num_results=5,
        grace_period=grace_period,
        mode="min"
    )    
    trainer = TorchTrainer(
        train_loop_per_worker=train_model,
        train_loop_config=config,
        scaling_config=scaling_config,
        run_config=RunConfig(
                name=model_name,
                storage_path=storage_path,
                stop=stopper
            )
    )

    result = trainer.fit()
    best_checkpoint = result.get_best_checkpoint("val_loss","min")

    print(f"Training result: {result}")
    print(f"Best checkpoint: {best_checkpoint}")

    # Create dir for saving model
    dir_path = os.path.join(work_dir,"trained_models/",model_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    # Save the best model
    try:
        best_checkpoint.to_directory(dir_path)        
    except Exception as e:
        print(f"Couldn't save the model because of {e}")

    # Plot graphs and save data
    try:
        history_path = os.path.join(result.path,"progress.csv")
        params_path = os.path.join(result.path,"params.json")
        print("history path found: ",history_path)

        # Load stuff
        results = pd.read_csv(history_path)
        with open(params_path,"r") as f:
            params = json.load(f)
        try:
            params = params["train_loop_config"]
        except:
            pass

        metrics_head = results.keys()[:10]
        history = {key:list(results[key]) for key in metrics_head}
        # history["params"] = params
        
        full_config = CONFIG
        full_config.update(params)
        full_config = rename_classes(full_config)

        # Dump history of the best trial
        json_dump(f'{dir_path}/history.json',history)
        json_dump(f'{dir_path}/params.json',full_config)
        # plot_training_graphs(history,dir_path) # TODO: plot graphs
        plot_training_history(history, dir_path)
    except Exception as e:
        print(f"Couldn't save history and plot graphs because of {e}")


def rename_classes(config:dict):
    for key, value in config.items():
        if 'class' in str(value):
            config[key] = str(value).split('.')[-1].strip(">'")
    return config


def train_model(train_config: dict):
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # generator = torch.Generator().manual_seed(seed_value)

    # load setup config and merge with training params
    config = CONFIG
    config.update(train_config)
    print("Training Config: ", config)

    # Set to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('CUDA availability:', device)
    
    # Get config for models
    hidden_channels=config.get("hidden_channels",32)
    out_channels= config.get("out_channels",16)
    layers = config.get("num_layers",4)
    layer_type=config.get("layer_type",GCNConv)
    scale_grad_by_freq = config.get("scale_grad_by_freq",True)
    latex_set = config.get("latex_set","OleehyO")
    vocab_type = config.get("vocab_type","concat")
    method = config.get("method",None)
    batch_norm = config.get("batch_norm",False)
    shuffle = config.get("shuffle",False)
    debug = config.get("debug",False)
    xml_name = config.get("xml_name", "debug")
    force_reload = config.get("force_reload", True)
    epochs = config.get("num_epochs",200)
    max_num_nodes = config.get("max_num_nodes",40)
    # sample_edges = config.get("sample_edges","sparse")
    sparse_edges = config.get("gen_sparse_edges",True)
    train_edge_features = config.get("train_edge_features",False)

    mn_type = config.get("mn_type","embed")

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
        # for component in ["tag","concat","pos","combined","mn","mi","mo","mtext"]:
        for component in ["tag","concat","pos","combined","split"]:
            dim = config.get(f"{component}_dim",None)
            if dim is not None:
                if component == "split":
                    method[embed].update({
                        "mi":dim,
                        "mo":dim,
                        "mtext":dim,
                        "mn":dim,
                    })
                    # if mn_type == "linear":
                    #     method["linear"] = {"mn":dim}
                    # else:
                    #     method["embed"] = {"mn":dim}
                else:
                    method[embed].update({component:dim})
        
    
    print("The embedding method is: ",method)
        

    print("Loading dataset...")

    # load and setup dataset
    mathml = MathmlDataset(xml_name,latex_set=latex_set,debug=debug)
    vocab = VocabBuilder(xml_name,vocab_type=vocab_type, debug=debug, reload_vocab=force_reload)
    dataset = GraphDataset(mathml.xml_dir,vocab, max_num_nodes= max_num_nodes, force_reload=force_reload, debug=debug)

    train, val, _ = dataset.split(shuffle=shuffle)
    
    print(f"MathML dataset size: {len(mathml)} equations")
    print(f"Dataset sizes: train= {len(train)} graphs, val= {len(val)} graphs. Max number of nodes per graph= {max_num_nodes}")

    # load models
    embedding_dim = sum(method["onehot"].values()) + sum(method["embed"].values()) + sum(method["linear"].values())

    encoder = GraphEncoder(embedding_dim,hidden_channels,out_channels,layers,layer_type,batch_norm)
    decoder = GraphDecoder(embedding_dim,hidden_channels,out_channels,layers,layer_type, edge_dim=1,batch_norm=batch_norm)
    model = GraphVAE(encoder, decoder, vocab.shape(), method, scale_grad_by_freq, sparse_edges, train_edge_features)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr",0.001))
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    model.to(device)

    
    train_loader = DataLoader(train, batch_size=config.get("batch_size",256), shuffle=True, num_workers=8)
    val_loader = DataLoader(val, batch_size=config.get("batch_size",256), shuffle=False, num_workers=8)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)
    # print("Number of nodes per batch: ", train_loader.dataset.))

    # Get checkpoint
    start = 0
    checkpoint = session.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            print("Found Checkpoints at: ", checkpoint_dir)
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            start = checkpoint_dict["epoch"] + 1
            model.load_state_dict(checkpoint_dict["model_state"])


    # TRAINING LOOP
    print("Starting training...")
    for epoch in range(start,epochs):

        train_loss, train_auc, train_ap, train_acc, train_sim = train_one_epoch(model,optimizer,train_loader,device,config)
        val_loss, val_auc, val_ap, val_acc, val_sim = validate(model,val_loader,device,config)

        # reduce learning rate
        scheduler.step(val_loss)

        metrics = {
            "loss": train_loss, "train_auc": train_auc, "train_ap": train_ap, "train_acc": train_acc, "train_sim": train_sim, 
            "val_loss": val_loss, "val_auc": val_auc, "val_ap": val_ap, "val_acc": val_acc, "val_sim": val_sim
        }
        
        # session.report(metrics)
        with tempfile.TemporaryDirectory() as tempdir:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict()},
                os.path.join(tempdir, "checkpoint.pt"),
            )
            session.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))


def train_one_epoch(model:GraphVAE,optimizer,train_loader,device, config ): # variational=False,force_undirected=True,neg_sampling_method="sparse"
    # Training phase
    model.train()
    total_train_loss = 0
    total_acc = 0
    total_sim = 0
    total_auc = 0
    total_ap = 0

    # Getting params
    variational = config.get("variational",False)
    force_undirected = config.get("force_undirected",True)
    neg_sampling_method = config.get("sample_edges","sparse")
    alpha = config.get("alpha",1)
    beta = config.get("beta",0)
    gamma = config.get("gamma",0)
    train_edge_features = config.get("train_edge_features",False)
    
    for i,batch in enumerate(train_loader):
        # if i % 2 == 0:
        #     print(f"Training batch #{i}..")
        
        optimizer.zero_grad()
        batch = batch.to(device)

        # Generate negative edges
        # pos_edge_index = batch.edge_index.to(device)
        num_edges = batch.num_nodes**2 if neg_sampling_method == "dense" else batch.edge_index.size(1)
        neg_edge_index = negative_sampling(
            edge_index=batch.edge_index, 
            num_nodes=batch.num_nodes, 
            num_neg_samples=num_edges,
            force_undirected=force_undirected,
            method=neg_sampling_method
        ).to(device)
        edge_weight = batch.edge_attr.to(device) if train_edge_features else None

        x = model.embed_x(batch.x,batch.tag,batch.pos,batch.nums).to(device)         
        z = model.encode(x, batch.edge_index, edge_weight)

        # Loss calculation
        loss = model.recon_full_loss(z, x, batch.edge_index, neg_edge_index, edge_weight, alpha, beta, gamma)
        if variational:
            loss = loss + (1 / batch.num_nodes) * model.kl_loss()

        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        # AUC, AP
        auc, ap = model.test(z, batch.edge_index, neg_edge_index)
        total_auc += auc
        total_ap += ap  

        # Accuracy and similarity
        acc, sim = model.test_nodes(z, batch.edge_index, x, batch.x, edge_weight)
        total_acc += acc
        total_sim += sim
    
    avg_train_loss = total_train_loss / len(train_loader)
    avg_auc = total_auc / len(train_loader)
    avg_ap = total_ap / len(train_loader)
    avg_acc = total_acc / len(train_loader)
    avg_sim = total_sim / len(train_loader)

    return avg_train_loss, avg_auc, avg_ap, avg_acc, avg_sim

def validate(model:GraphVAE,val_loader,device,config): # variational=False,force_undirected=True,neg_sampling_method="sparse"
    model.eval()
    total_val_loss = 0
    total_auc = 0
    total_ap = 0
    total_acc = 0
    total_sim = 0


    # Getting params
    variational = config.get("variational",False)
    force_undirected = config.get("force_undirected",True)
    neg_sampling_method = config.get("sample_edges","sparse")
    alpha = config.get("alpha",1)
    beta = config.get("beta",0)
    gamma = config.get("gamma",0)
    train_edge_features = config.get("train_edge_features",False)

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            # Generate negative edges
            pos_edge_index = batch.edge_index
            num_edges = batch.num_nodes**2 if neg_sampling_method == "dense" else pos_edge_index.size(1)
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index, 
                num_nodes=batch.num_nodes, 
                num_neg_samples=num_edges,
                force_undirected=force_undirected,
                method=neg_sampling_method
            ).to(device)
            edge_weight = batch.edge_attr.to(device) if train_edge_features else None
            
            x = model.embed_x(batch.x,batch.tag,batch.pos,batch.nums).to(device)          
            z = model.encode(x, batch.edge_index,edge_weight)

            # Loss
            loss = model.recon_full_loss(z, x, pos_edge_index, neg_edge_index, edge_weight, alpha, beta, gamma)
            if variational:
                loss = loss + (1 / batch.num_nodes) * model.kl_loss()
            total_val_loss += loss.item()

            # AUC, AP
            auc, ap = model.test(z, pos_edge_index, neg_edge_index)
            total_auc += auc
            total_ap += ap  

            # Accuracy and similarity
            acc, sim = model.test_nodes(z, pos_edge_index,x,batch.x, edge_weight)
            total_acc += acc
            total_sim += sim

    avg_val_loss = total_val_loss / len(val_loader)
    avg_auc = total_auc / len(val_loader)
    avg_ap = total_ap / len(val_loader)
    avg_acc = total_acc / len(val_loader)
    avg_sim = total_sim / len(val_loader)


    return avg_val_loss, avg_auc, avg_ap, avg_acc, avg_sim


