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
from preprocessing.GraphEmbedder import GraphEmbedder, MATHML_TAGS
from models.Graph.GraphDataset import GraphDataset
from models.Graph.GraphAutoEncoder import Encoder
import random
# from tensorboardX import SummaryWriter

from utils.plot import plot_loss_graph, plot_training_graphs
from utils.save import json_dump
import pandas as pd
import tempfile





def main(model_name="VGAE",epochs=200):
    storage_path= "/data/nsam947/Freshwater-Modelling/data/ray_results"
    work_dir = "/data/nsam947/Freshwater-Modelling"
    trials_dir = os.path.join(storage_path, model_name)
    # checkpoint_dir = os.path.join(trials_dir, "checkpoints")
    print("Current Working Directory:", os.getcwd())



    config = {
        "model_name": model_name,
        "num_epochs": epochs,
        "lr": 1e-3,
        "variational": True
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
    seed_value = 0
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load setup config and merge with training params
    config = CONFIG
    config.update(train_config)

    # Set to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('CUDA availability:', device)
    
    # load and setup dataset
    root_dir = "/data/nsam947/Freshwater-Modelling/dataset/"
    dataset = GraphDataset(root=root_dir, equations_file='cleaned_formulas_katex.xml',force_reload=False)  

    data_size = len(dataset)
    train_data = dataset[:int(data_size * 0.8)]
    val_data = dataset[int(data_size * 0.8):int(data_size * 0.9)]

    # Get channels for layers
    in_channels = dataset.num_features
    vocab_size = dataset.vocab_size

    # load model
    encoder = Encoder(
        in_channels,
        hidden_channels=config.get("hidden_channels",32),
        out_channels= config.get("out_channels",16),
        layers = config.get("num_layers",4), 
        embedding_dim=config.get("embedding_dims",200),
        vocab_size=vocab_size,
        scale_grad_by_freq=config.get("scale_grad_by_freq",True),
        layer_type=config.get("layer_type",GCNConv), 
        variational=config.get("variational",False),
        batch_norm=config.get("batch_norm",False)
    )
    model = pyg_nn.VGAE(encoder) if config.get("variational",False) else pyg_nn.GAE(encoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr",0.001))
    model.to(device)

    
    train_loader = DataLoader(train_data, batch_size=config.get("batch_size",256), shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=config.get("batch_size",256), shuffle=False, num_workers=8)
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
    for epoch in range(start,config.get("num_epochs",500)):
        # Train phase
        avg_train_loss = train_one_epoch(
            model,optimizer,epoch,train_loader,device,
            searching=True,
            neg_sampling_method=config.get("sample_edges","sparse"),
            variational=config.get("variational",False),
            force_undirected=config.get("force_undirected",True)
        )

        # Validation phase
        avg_val_loss, avg_auc, avg_ap = validate(
            model,val_loader,device,
            neg_sampling_method=config.get("sample_edges","sparse"),
            variational=config.get("variational",False),
            force_undirected=config.get("force_undirected",True)
        )

        metrics = {"loss": avg_train_loss, "val_loss": avg_val_loss, "auc":avg_auc,"ap":avg_ap}
        
        # session.report(metrics)
        with tempfile.TemporaryDirectory() as tempdir:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict()},
                os.path.join(tempdir, "checkpoint.pt"),
            )
            session.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))


def train_one_epoch(model,optimizer,epoch,train_loader,device, searching=False,variational=False,force_undirected=True,neg_sampling_method="sparse"):
    # Training phase
    model.train()
    total_train_loss = 0

    # # Setup tqdm or not if searching
    # if searching:
    #     batches = train_loader
    # else:
    #     batches = tqdm(train_loader,desc=f"training epoch {epoch}",unit="batch")
    
    for batch in train_loader:
        optimizer.zero_grad()
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
        
        z = model.encode(batch.x, batch.edge_index)
        loss = model.recon_loss(z, pos_edge_index, neg_edge_index)

        if variational:
            loss = loss + (1 / batch.num_nodes) * model.kl_loss()

        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    return avg_train_loss

def validate(model,val_loader,device,variational=False,force_undirected=True,neg_sampling_method="sparse"):
    model.eval()
    total_val_loss = 0
    total_auc = 0
    total_ap = 0

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
            
            z = model.encode(batch.x, batch.edge_index)
            loss = model.recon_loss(z, pos_edge_index, neg_edge_index)

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

    

# def main():
#     # Fix the randomness
#     seed_value = 0
#     random.seed(seed_value)
#     np.random.seed(seed_value)
#     torch.manual_seed(seed_value)
#     torch.cuda.manual_seed(seed_value)
#     torch.cuda.manual_seed_all(seed_value)  # if using multiple GPUs
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


#     # # path = "/data/nsam947/ray_results/train_model_2024-06-05_18-10-34/train_model_2271f6c6_20_batch_size\=256\,hidden_channels\=32\,lr\=0.0011\,out_channels\=16_2024-06-05_18-50-55/result.json"
#     # result_dir = "/data/nsam947/ray_results/train_model_2024-06-05_18-10-34/"
#     # trials = os.listdir(result_dir)
#     # trial_path = [path for path in trials if "2271f6c6" in path][0]
#     # trial_path = os.path.join(result_dir,trial_path,"progress.csv")

#     # result = pd.read_csv(trial_path)
#     # # with open(trial_path,"r") as f:
#     # #     result = json.load(f)
    
#     # metrics_head = result.keys()[:4]
#     # metrics = {key:list(result[key]) for key in metrics_head}
#     # plot_training_graphs(metrics,"out/")

#     # return

#     # Create dir for saving model
#     dir_path = "trained_models/exp_5"
#     if not os.path.exists(dir_path):
#         os.mkdir(dir_path)

#     # Setup device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print('CUDA availability:', device)

#     # Load dataset 
#     print("Loading dataset...")
#     # data augm
#     # transform = T.Compose([
#     #     # T.NormalizeFeatures(),
#     #     T.ToDevice(device),
#     #     # T.RandomNodeSplit()
#     #     # T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True, # splits on the graph level, and in my case I have a list of graphs
#     #     #     split_labels=False, add_negative_train_samples=False),
#     # ])
#     dataset = GraphDataset(root='dataset/', equations_file='cleaned_formulas_katex.xml',force_reload=False, debug=False) #transform=transform)
#     in_channels = dataset.num_features
#     vocab_size = dataset.vocab_size


#     print(dataset.get_summary())
#     print("Number of input channels: ", in_channels)
    
#     # return

#     data_size = len(dataset)
#     train_data = dataset[:int(data_size * 0.8)]
#     val_data = dataset[int(data_size * 0.8):int(data_size * 0.9)]
#     test_data = dataset[int(data_size * 0.9):]

#     # model = pyg_nn.VGAE(VariationalGCNEncoder(in_channels, 16))
#     # model = pyg_nn.GAE(GraphSAGE(in_channels,hidden_channels=32,num_layers=4,out_channels=4))
#     model = pyg_nn.GAE(Encoder(in_channels,16,8,embedding_dim=192,layers = 4,vocab_size=vocab_size,scale_grad_by_freq=False))
#     # model = pyg_nn.GAE(GraphEncoder(in_channels, 16,8))
#     variational=False


#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
#     # criterion = torch.nn.MSELoss()

#     # Early stopping
#     patience = 5  # Number of epochs to wait for improvement before stopping
#     patience_counter = 0
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=patience, threshold=0.01)

#     writer = SummaryWriter("log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),comment="training")

#     # Put model to device
#     model = model.to(device)

#     batch_size = 128
#     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
#     val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)
#     test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)

#     # Training loop
#     best_val_loss = float('inf')
#     num_epochs = 25
#     history = {"loss":[],"val_loss":[],"auc":[],"ap":[]}

#     # Training loop
#     for epoch in range(num_epochs):
#         # Training phase
#         avg_train_loss = train_one_epoch(model,optimizer,epoch,train_loader,device, variational=variational)
        
#         # validation phase
#         avg_val_loss, avg_auc, avg_ap = validate(model,val_loader,device,variational=variational)
        
#         # Add to tensorboardX
#         # writer.add_scalar("Loss/train", avg_train_loss, epoch)
#         # writer.add_scalar("Loss/val", avg_val_loss, epoch)
#         writer.add_scalars("Losses",{"train":avg_train_loss,"val":avg_val_loss},epoch)
#         writer.add_scalar("Metrics/AUC", avg_auc, epoch)
#         writer.add_scalar("Metrics/AP", avg_ap, epoch)

#         history["loss"].append(avg_train_loss)
#         history["val_loss"].append(avg_val_loss)
#         history["auc"].append(avg_auc)
#         history["ap"].append(avg_ap)

#         print(f'Epoch: {epoch:03d}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, AUC: {avg_auc:.4f}, AP: {avg_ap:.4f}')
        
#         # Scheduler step
#         scheduler.step(avg_val_loss)
        
#         # Early stopping
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             patience_counter = 0
#         else:
#             patience_counter += 1
        
#         if patience_counter >= patience:
#             print("Early stopping triggered")
#             break
    
    

#     # Save the best model
#     try:
#         torch.save(model.state_dict(), f'{dir_path}/model_weights.pt')
#     except Exception as e:
#         print(f"Couldn't save the model because of {e}")

#     # Plot graphs and save data
#     try:
#         plot_training_graphs(history,dir_path)
#         json_dump(f'{dir_path}/history.json',history)
#     except Exception as e:
#         print(f"Couldn't save history and plot graphs because of {e}")

#     # Load the best model (if early stopping was triggered)
#     try:
#         model.load_state_dict(torch.load(f'{dir_path}/model_weights.pt'))
#     except Exception as e:
#         print(f"Couldn't load the model because of {e}")

#     test_model(model,test_loader,device)