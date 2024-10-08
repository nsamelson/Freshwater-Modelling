import json
import os
import random
import numpy as np
import pandas as pd
from ray import tune, train
from ray.train import Checkpoint, RunConfig
from ray.air import session, config
from ray.tune.stopper import TrialPlateauStopper, ExperimentPlateauStopper, CombinedStopper
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import torch
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GraphSAGE, GCNConv, GraphConv
from torch_geometric.loader import DataLoader

from config import CONFIG
from models.Graph.GraphAutoEncoder import GraphEncoder, GraphDecoder, GraphVAE
from models.train import train_model
from preprocessing.GraphDataset import GraphDataset
from utils import stats
from utils import plot
from utils.plot import plot_training_graphs, plot_hyperparam_search, plot_training_history
from utils.stats import extract_data_from_search
from utils.save import json_dump
import tempfile




def main(model_name, latex_set, vocab_type, xml_name,max_num_epochs=50, num_samples=50,gpus_per_trial=float(1/4)):

    storage_path= "/data/nsam947/Freshwater-Modelling/data/ray_results"
    work_dir = "/data/nsam947/Freshwater-Modelling"
    tmp_dir = "/data/nsam947/tmp"

    os.makedirs(tmp_dir, exist_ok=True)
    os.chmod(tmp_dir, 0o777)  # Adjust permissions as needed
    os.environ["RAY_TMPDIR"] = tmp_dir
    
    trials_dir = os.path.join(storage_path, model_name)
    # checkpoint_dir = os.path.join(trials_dir, "checkpoints")
    print("Current Working Directory:", os.getcwd())

    # Parameters to tune
    search_space = {
        "num_epochs":max_num_epochs,
        "latex_set":latex_set,
        "vocab_type":vocab_type,
        "xml_name":xml_name,
        "model_name":model_name,

        # "concat_dim":256,
        # "combined_dim":256,
        # "embed_method": "onehot",
        "tag_dim": 256,
        "train_edge_features":True,
        "pos_dim":None,
        "split_dim": 256,
        "embed_method":"freq_embed",
        "num_layers":2,
        "out_channels":32,
        "hidden_channels":512,


        # "alpha": tune.qloguniform(0.5, 1,5e-2),
        # "beta": tune.qloguniform(0.5, 1,5e-2),
        # "gamma": tune.qloguniform(0.5, 1,5e-2),        
        "lr": tune.qloguniform(1e-5, 1e-2,5e-6),
        "batch_size": tune.choice([64, 128, 256, 512, 1024]),
        # "variational": tune.grid_search([True,False]),
        # "mn_type": tune.grid_search(["embed","linear"]),
        # "mn_dim": tune.grid_search([None,256]),
        # "mi_dim": tune.grid_search([None,256]),
        # "mo_dim": tune.grid_search([None,256]),
        # "mtext_dim": tune.grid_search([None,256]),
        # "train_edge_features": tune.grid_search([True,False]),
        # "split_dim":tune.grid_search([64,256,512,1024]),
        # "pos_dim": tune.grid_search([None,256]),
        # "tag_dim": tune.grid_search([None,256]),
        # "concat_dim": tune.grid_search([256,512,1024]),
        # "loss": tune.grid_search(["mse","cosine"]),
        # "embed_method": tune.grid_search(["onehot","freq_embed"])
        # "scale_grad_by_freq": tune.grid_search([True,False]),
        # "num_layers": tune.grid_search([2,3,4,5,6]),
        # "hidden_channels": tune.grid_search([32,64,128,256,512]),
        # "out_channels": tune.grid_search([8,16,32,64]),
        # "embedding_dims":tune.choice([10,50,100,200,500,1000]),
        # "layer_type": tune.choice([GCNConv, GraphConv]),
  
    }

    hyperopt_search = HyperOptSearch(metric="val_acc", mode="max")
    grace_period = 10

    # Define ASHA scheduler
    asha_scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=max_num_epochs,   # Maximum number of training iterations
        grace_period=grace_period,         # Number of iterations before considering early stopping
        reduction_factor=3,      # keeps the top 1/reduction_factor running, the rest is pruned
        brackets=2              # more brackets = more exploration in the parameters
    )

    # Early stopper
    stopper = TrialPlateauStopper(
        metric="val_loss",
        num_results=5,
        grace_period=grace_period,
        mode="min"
    )

    # Restore or run a new tuning
    if tune.Tuner.can_restore(trials_dir):
        tuner = tune.Tuner.restore(
            trials_dir, 
            trainable=tune.with_resources(
                tune.with_parameters(train_model),
                resources={"cpu": 8, "gpu": gpus_per_trial}
            ), 
            resume_errored=True,
            param_space= search_space
        )
    else:
        tuner = tune.Tuner(
            trainable=tune.with_resources(
                tune.with_parameters(train_model),
                resources={"cpu": 8, "gpu": gpus_per_trial}
            ),
            tune_config=tune.TuneConfig(
                search_alg=hyperopt_search,  
                scheduler=asha_scheduler,  
                num_samples=num_samples,  # Adjust based on your budget
            ),
            param_space=search_space,
            run_config=RunConfig(
                name=model_name,
                storage_path=storage_path,
                failure_config=config.FailureConfig(max_failures=-1),
                stop=stopper
            )
        )

    results = tuner.fit()
    best_result = results.get_best_result("val_acc", "max","all")
    best_model = best_result.get_best_checkpoint("val_acc","max")

    # Print the best results
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(best_result.metrics["val_loss"]))

    # Create dir for saving model
    dir_path = os.path.join(work_dir,"trained_models/",model_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    # else:
    #     os.mkdir(os.path.join(dir_path,"_1"))

    # Save the best model
    try:
        best_model.to_directory(dir_path)        
    except Exception as e:
        print(f"Couldn't save the model because of {e}")

    # Plot graphs and save data
    try:
        history_path = os.path.join(best_result.path,"progress.csv")
        params_path = os.path.join(best_result.path,"params.json")
        print("history path found: ",history_path)

        # Load stuff
        results = pd.read_csv(history_path)
        with open(params_path,"r") as f:
            params = json.load(f)

        metrics_head = results.keys()[:10]
        history = {key:list(results[key]) for key in metrics_head}
        # history["params"] = params

        full_config = CONFIG
        full_config.update(params)
        full_config = rename_classes(full_config)
        # Dump history of the best trial
        json_dump(f'{dir_path}/history.json',history)
        json_dump(f'{dir_path}/params.json',full_config)

        # # Dump history of the best trial
        # json_dump(f'{dir_path}/history.json',history)
        # plot_training_graphs(history,dir_path)
        plot_training_history(history, dir_path)
    except Exception as e:
        print(f"Couldn't save history and plot graphs because of {e}")

    # Plot the hyperparam search
    try:
        extract_data_from_search(trials_dir)
        stats.create_combined_dataframe(trials_dir,f"trained_models/{model_name}")
        # plot_hyperparam_search(dir_path)
        plot.plot_boxplot_hyperparameters(model_name)
    except Exception as e:
        print(f"Couldn't create boxplot of hyperparams search")

def rename_classes(config:dict):
    for key, value in config.items():
        if 'class' in str(value):
            config[key] = str(value).split('.')[-1].strip(">'")
    return config


# def train_model(config={}):
#     seed_value = 0
#     random.seed(seed_value)
#     np.random.seed(seed_value)
#     torch.manual_seed(seed_value)
#     torch.cuda.manual_seed(seed_value)
#     torch.cuda.manual_seed_all(seed_value)  # if using multiple GPUs
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

#     # Get stuff out of config
#     num_epochs = config.get("num_epochs",500)
#     lr = config.get("lr",0.001)
#     layer_type = config.get("layer_type",GCNConv)
#     batch_size = config.get("batch_size",256)
#     num_layers = config.get("num_layers",4)
#     out_channels=config.get("out_channels",16)
#     hidden_channels = config.get("hidden_channels",2*out_channels)
#     embedding_dims = config.get("embedding_dims",200)
#     scale_grad_by_freq = config.get("scale_grad_by_freq",True)
#     sample_edges = config.get("sample_edges","sparse")
#     variational = config.get("variational",False)
#     batch_norm = config.get("batch_norm",False)
#     force_undirected = config.get("force_undirected",True)


#     # Set to device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print('CUDA availability:', device)
    
#     # load and setup dataset
#     root_dir = "/data/nsam947/Freshwater-Modelling/dataset/"
#     dataset = GraphDataset(root=root_dir, equations_file='cleaned_formulas_katex.xml',force_reload=False)  

#     data_size = len(dataset)
#     train_data = dataset[:int(data_size * 0.8)]
#     val_data = dataset[int(data_size * 0.8):int(data_size * 0.9)]

#     # Get channels for layers
#     in_channels = dataset.num_features
#     vocab_size = dataset.vocab_size

#     # load model
#     encoder = Encoder(
#         in_channels,hidden_channels,out_channels,
#         layers = num_layers, 
#         embedding_dim=embedding_dims,
#         vocab_size=vocab_size,
#         scale_grad_by_freq=scale_grad_by_freq,
#         layer_type=layer_type, 
#         variational=variational,
#         batch_norm=batch_norm
#     )
#     model = pyg_nn.VGAE(encoder) if variational else pyg_nn.GAE(encoder)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     model.to(device)

    
#     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
#     val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)
#     # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)

#     # Get checkpoint
#     start = 0
#     checkpoint = session.get_checkpoint()
#     if checkpoint:
#         with checkpoint.as_directory() as checkpoint_dir:
#             print("Found Checkpoints at: ", checkpoint_dir)
#             checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
#             start = checkpoint_dict["epoch"] + 1
#             model.load_state_dict(checkpoint_dict["model_state"])

#     for epoch in range(start,num_epochs):
#         # Train phase
#         avg_train_loss = train_one_epoch(model,optimizer,epoch,train_loader,device,searching=True,neg_sampling_method=sample_edges,variational=variational,force_undirected=force_undirected)
#         avg_val_loss, avg_auc, avg_ap = validate(model,val_loader,device,neg_sampling_method=sample_edges,variational=variational,force_undirected=force_undirected)

#         metrics = {"loss": avg_train_loss, "val_loss": avg_val_loss, "auc":avg_auc,"ap":avg_ap}
        
#         # session.report(metrics)
#         with tempfile.TemporaryDirectory() as tempdir:
#             torch.save(
#                 {"epoch": epoch, "model_state": model.state_dict()},
#                 os.path.join(tempdir, "checkpoint.pt"),
#             )
#             session.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))
