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

from utils.plot import plot_loss_graph, plot_training_graphs
from utils.save import json_dump
import pandas as pd
import tempfile





def main(model_name="GraphVAE_new",epochs=200):
    storage_path= "/data/nsam947/Freshwater-Modelling/data/ray_results"
    work_dir = "/data/nsam947/Freshwater-Modelling"
    trials_dir = os.path.join(storage_path, model_name)
    # checkpoint_dir = os.path.join(trials_dir, "checkpoints")
    print("Current Working Directory:", os.getcwd())



    config = {
        "model_name": model_name,
        "num_epochs": epochs,
        "lr": 1e-3,
        "variational": True,
        "alpha": 1,
        "beta": 1,
        "gamma":1,
        "latex_set":"OleehyO",
        "vocab_type":"concat",
        "embedding_dim":256,
        "method":{
            "onehot": ["concat"],
            "embed": [], # "mi","mo","mtext","mn"
            "linear": [],
            "scale": "log"
        }
    }


    grace_period = 15 if epochs != 1 else 1

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

        metrics_head = results.keys()[:4]
        history = {key:list(results[key]) for key in metrics_head}
        # history["params"] = params
        
        full_config = CONFIG
        full_config.update(params)
        full_config = rename_classes(full_config)

        # Dump history of the best trial
        json_dump(f'{dir_path}/history.json',history)
        json_dump(f'{dir_path}/params.json',full_config)
        plot_training_graphs(history,dir_path)
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
    
    # root_dir = "/data/nsam947/Freshwater-Modelling/dataset/"
    # dataset = GraphDataset(root=root_dir, equations_file='cleaned_formulas_katex.xml',force_reload=False)  

    # data_size = len(dataset)
    # train_data = dataset[:int(data_size * 0.8)]
    # val_data = dataset[int(data_size * 0.8):int(data_size * 0.9)]

    # Get config for models
    embedding_dim = config.get("embedding_dims",200)
    # in_channels = dataset.num_features + embedding_dim - 1
    hidden_channels=config.get("hidden_channels",32)
    out_channels= config.get("out_channels",16)
    layers = config.get("num_layers",4)
    layer_type=config.get("layer_type",GCNConv)
    scale_grad_by_freq = config.get("scale_grad_by_freq",True)
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
    

    # load models
    # encoder = GraphEncoder(in_channels,hidden_channels,out_channels,layers,layer_type)
    # decoder = GraphDecoder(in_channels,hidden_channels,out_channels,layers,layer_type)
    # model = GraphVAE(encoder, decoder, dataset.vocab_size, embedding_dim, scale_grad_by_freq)

    encoder = GraphEncoder(embedding_dim,hidden_channels,out_channels,layers,layer_type)
    decoder = GraphDecoder(embedding_dim,hidden_channels,out_channels,layers,layer_type)
    model = GraphVAE(encoder, decoder, vocab.shape(), embedding_dim,method ,scale_grad_by_freq)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr",0.001))
    model.to(device)

    
    train_loader = DataLoader(train, batch_size=config.get("batch_size",256), shuffle=True, num_workers=8)
    val_loader = DataLoader(val, batch_size=config.get("batch_size",256), shuffle=False, num_workers=8)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)

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
    for epoch in range(start,config.get("num_epochs",500)):

        avg_train_loss = train_one_epoch(model,optimizer,train_loader,device,config)
        avg_val_loss, avg_auc, avg_ap = validate(model,val_loader,device,config)

        metrics = {"loss": avg_train_loss, "val_loss": avg_val_loss, "auc":avg_auc,"ap":avg_ap}
        
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

    # Getting params
    variational = config.get("variational",False)
    force_undirected = config.get("force_undirected",True)
    neg_sampling_method = config.get("neg_sampling_method","sparse")
    alpha = config.get("alpha",1)
    beta = config.get("beta",0)
    gamma = config.get("gamma",0)
    
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)

        # Generate negative edges
        pos_edge_index = batch.edge_index.to(device)
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index, 
            num_nodes=batch.num_nodes, 
            num_neg_samples=pos_edge_index.size(1),
            force_undirected=force_undirected,
            method=neg_sampling_method
        ).to(device)

        x = model.embed_x(batch.x,batch.tag_index,batch.pos,batch.nums).to(device)         
        z = model.encode(x, batch.edge_index, batch.edge_attr)
        loss = model.recon_full_loss(z, x, pos_edge_index, neg_edge_index, batch.edge_attr, alpha, beta, gamma)

        if variational:
            loss = loss + (1 / batch.num_nodes) * model.kl_loss()

        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    return avg_train_loss

def validate(model:GraphVAE,val_loader,device,config): # variational=False,force_undirected=True,neg_sampling_method="sparse"
    model.eval()
    total_val_loss = 0
    total_auc = 0
    total_ap = 0

    # Getting params
    variational = config.get("variational",False)
    force_undirected = config.get("force_undirected",True)
    neg_sampling_method = config.get("neg_sampling_method","sparse")
    alpha = config.get("alpha",1)
    beta = config.get("beta",0)
    gamma = config.get("gamma",0)

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            # Generate negative edges
            pos_edge_index = batch.edge_index
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index, 
                num_nodes=batch.num_nodes, 
                num_neg_samples=pos_edge_index.size(1),
                force_undirected=force_undirected,
                method=neg_sampling_method
            ).to(device)
            
            # x = model.embed_x(batch.x)        
            # z = model.encode(x, batch.edge_index)
            # # loss = model.recon_loss(z, pos_edge_index, neg_edge_index)
            # loss = model.recon_full_loss(z,x,pos_edge_index, neg_edge_index,alpha,beta)
            x = model.embed_x(batch.x,batch.tag_index,batch.pos,batch.nums).to(device)          
            z = model.encode(x, batch.edge_index, batch.edge_attr)
            loss = model.recon_full_loss(z, x, pos_edge_index, neg_edge_index, batch.edge_attr, alpha, beta, gamma)

            if variational:
                loss = loss + (1 / batch.num_nodes) * model.kl_loss()

            total_val_loss += loss.item()
            auc, ap = model.test(z, pos_edge_index, neg_edge_index)
            
            total_auc += auc
            total_ap += ap    

    avg_val_loss = total_val_loss / len(val_loader)
    avg_auc = total_auc / len(val_loader)
    avg_ap = total_ap / len(val_loader)

    return avg_val_loss, avg_auc, avg_ap


