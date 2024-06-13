import os
import random
import numpy as np
from ray import tune 
from ray.train import Checkpoint, RunConfig
from ray.air import session
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import torch
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GraphSAGE, GCN
from torch_geometric.loader import DataLoader

from models.Graph.GraphAutoEncoder import Encoder, GraphEncoder
from models.train import train_one_epoch, validate
from models.Graph.GraphDataset import GraphDataset
from utils.plot import plot_training_graphs
from utils.save import json_dump

def main(num_samples=1,max_num_epochs=30,gpus_per_trial=1,debug=False):

    storage_path= "/data/nsam947/Freshwater-Modelling/data/ray_results"
    model_name = "GAE_search_1"
    work_dir = "/data/nsam947/Freshwater-Modelling"
    trials_dir = os.path.join(storage_path,model_name)

    print("Current Working Directory:", os.getcwd())

    # Parameters to tune
    search_space = {
        # "lr": tune.loguniform(1e-5, 1e-1),
        # "batch_size": tune.choice([32, 64, 128, 256]),
        # "encoder": tune.choice(['GCN', 'GAE', 'GraphSAGE']),
        # "decoder": tune.choice([None]),
        # "num_layers": tune.choice([1,2,3,4,5]),
        # "hidden_channels": tune.choice([16,32,64,128,256]),
        "out_channels": tune.grid_search([8,16,32]),
        "embedding_dims":tune.grid_search([10,25,50])
    }

    hyperopt_search = HyperOptSearch(metric="val_loss", mode="min")
    # hyperband_scheduler = HyperBandScheduler(metric="val_loss", mode="min")
    # Define ASHA scheduler
    asha_scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=max_num_epochs,   # Maximum number of training iterations
        grace_period=5,         # Number of iterations before considering early stopping
        reduction_factor=2      # Reduction factor for successive halving
    )

    # Run the tuning
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model),
            resources={"cpu": 8, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            search_alg=hyperopt_search,  
            scheduler=asha_scheduler,  
            num_samples=num_samples,  # Adjust based on your budget
        ),
        param_space=search_space,
        # run_config=RunConfig(model_name,storage_path)
    )

    results = tuner.fit()
    best_result = results.get_best_result("val_loss", "min")

    # Print the best results
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(best_result.metrics["loss"]))

    # Create dir for saving model
    dir_path = os.path.join(work_dir,"trained_models/search_2")
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    # Save the best model
    try:
        torch.save(best_result.state_dict(), f'{dir_path}/model_weights.pt')
    except Exception as e:
        print(f"Couldn't save the model because of {e}")

    # # Plot graphs and save data
    # try:
    #     plot_training_graphs(history,dir_path)
    #     json_dump(f'{dir_path}/history.json',history)
    # except Exception as e:
    #     print(f"Couldn't save history and plot graphs because of {e}")

def train_model(config):
    seed_value = 0
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Get stuff out of config
    out_channels=config.get("out_channels",8)
    hidden_channels = config.get("hidden_channels",2*out_channels)
    embedding_dims = config.get("embedding_dims",10)
    lr = config.get("lr",0.001)
    batch_size = config.get("batch_size",128)


    root_dir = "/data/nsam947/Freshwater-Modelling/dataset/"

    # Set to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('CUDA availability:', device)
    
    # load and setup dataset
    dataset = GraphDataset(root=root_dir, equations_file='cleaned_formulas_katex.xml',force_reload=True, embedding_dims=embedding_dims) 
    data_size = len(dataset)
    train_data = dataset[:int(data_size * 0.8)]
    val_data = dataset[int(data_size * 0.8):int(data_size * 0.9)]
    # test_data = dataset[int(data_size * 0.9):]

    # Get channels for layers
    in_channels = dataset.num_features

    # load model
    model = pyg_nn.GAE(Encoder(in_channels, hidden_channels,out_channels))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)

    for epoch in range(50):
        # Train phase
        avg_train_loss = train_one_epoch(model,optimizer,epoch,train_loader,device,searching=True)
        
        # Validation phase
        avg_val_loss, avg_auc, avg_ap = validate(model,val_loader,device)

        metrics = {"loss": avg_train_loss, "val_loss": avg_val_loss, "auc":avg_auc,"ap":avg_ap}
        
        session.report(metrics)
