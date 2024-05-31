import json
from matplotlib import pyplot as plt
import numpy as np
# from sklearn.base import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, InMemoryDataset, collate, Batch, Dataset
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import DataLoader
import torch_geometric.nn as pyg_nn
import xml.etree.ElementTree as ET
import torch_geometric.transforms as T
from tqdm import tqdm
from preprocessing.GraphEmbedder import GraphEmbedder, MATHML_TAGS
from models.Graph.GraphDataset import GraphDataset
from models.Graph.GraphAutoEncoder import Encoder, GraphAutoencoder, GraphEncoder
import random



def main():
    # Fix the randomness
    seed_value = 0
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('CUDA availability:', device)

    # Load dataset 
    print("Loading dataset...")
    dataset = GraphDataset(root='dataset/', equations_file='cleaned_formulas_katex.xml')
    # dataset.process(debug=False) # TODO: uncomment to reprocess if change of dataset or preprocessing!!!

    print(dataset.get_summary())
    data_size = len(dataset)
    train_data = dataset[:int(data_size * 0.8)]
    val_data = dataset[int(data_size * 0.8):]

    in_channels = dataset.num_features
    model = pyg_nn.GAE(Encoder(in_channels, 8))
    # model = pyg_nn.GAE(GraphEncoder(in_channels, 16,8))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Put model to device
    model = model.to(device)

    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)

    # Training loop
    best_val_loss = float('inf')
    patience = 10  # Number of epochs to wait for improvement before stopping
    patience_counter = 0


    # Training loop
    for epoch in range(20):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader,desc=f"training epoch {epoch}",unit="batch"):
            optimizer.zero_grad()
            batch = batch.to(device)

            # Generate negative edges
            pos_edge_index = batch.edge_index
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index, 
                num_nodes=batch.num_nodes, 
                num_neg_samples=pos_edge_index.size(1)
            ).to(device)
            
            z = model.encode(batch.x, batch.edge_index)
            loss = model.recon_loss(z, pos_edge_index, neg_edge_index)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        all_pos_preds = []
        all_neg_preds = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                # Generate negative edges
                pos_edge_index = batch.edge_index
                neg_edge_index = negative_sampling(
                    edge_index=pos_edge_index, 
                    num_nodes=batch.num_nodes, 
                    num_neg_samples=pos_edge_index.size(1)
                ).to(device)
                
                z = model.encode(batch.x, batch.edge_index)
                loss = model.recon_loss(z, pos_edge_index, neg_edge_index)
                total_val_loss += loss.item()

                pos_pred = model.decoder(z, pos_edge_index)
                neg_pred = model.decoder(z, neg_edge_index)

                all_pos_preds.append(pos_pred.cpu())
                all_neg_preds.append(neg_pred.cpu())
        
        avg_val_loss = total_val_loss / len(val_loader)
        all_pos_preds = torch.cat(all_pos_preds, dim=0)
        all_neg_preds = torch.cat(all_neg_preds, dim=0)
        y_pred = torch.cat([all_pos_preds, all_neg_preds], dim=0)
        y_true = torch.cat([torch.ones(all_pos_preds.size(0)), torch.zeros(all_neg_preds.size(0))], dim=0)

        auc = roc_auc_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)

        print(f'Epoch: {epoch:03d}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}')
        
        # Scheduler step
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'trained_models/best_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
    
    # Load the best model (if early stopping was triggered)
    model.load_state_dict(torch.load('trained_models/best_model.pth'))


    # dataset_path = "dataset/graph_formulas_katex_debug.pt"
    # # dataset_path = "dataset/graph_formulas_katex.pt"
    # dataset = torch.load(dataset_path)

    # return
    # Split sets
    # Split edges for each graph in the dataset
    # train_set, val_set, test_set = split_edges(dataset)
    # data_size = len(dataset)
    # train_set = dataset[:int(data_size * 0.8)]
    # val_set = dataset[int(data_size * 0.8):int(data_size * 0.9)]
    # test_set = dataset[int(data_size * 0.9):]


    # def collate_fn(data_list):
    #     batch = Batch.from_data_list(data_list)
    #     batch.x = embedder.embed_into_vector(batch.x)
    #     return batch

    # data = GraphDataset(dataset,embedder=embedder)
    # print(data)

    return
    # Create a DataLoader
    batch_size = 64
    embedder = GraphEmbedder(scale_by_freq=False)
    train_loader = EmbeddedDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, embedder=embedder)
    val_loader = EmbeddedDataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=8, embedder=embedder)
    test_loader = EmbeddedDataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=8, embedder=embedder)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=0)


    # Calculate # of channels: should be 67
    in_channels = len(MATHML_TAGS) + 4 * (embedder.embedding_dims - 1) 
    
    # return

    # model = GraphAutoencoder(in_channels,16,8)
    model = pyg_nn.GAE(Encoder(in_channels, 16))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    # Put model to device
    model = model.to(device)


    # Training loop
    for epoch in range(1, 10):
        model.train()
        train_loss = 0
        val_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_hat, z = model(batch)
            # z = model.encode(batch.x, batch.edge_index)
            # loss = model.recon_loss(z, batch.edge_index)
            loss = criterion(x_hat, batch.x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            

        # Validation step
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                x_hat, z = model(batch)
                loss = criterion(x_hat, batch.x)
                val_loss += loss.item()
                # z = model.encode(batch.x, batch.edge_index)
                # val_loss = model.recon_loss(z, batch.edge_index)
        print(f'Epoch: {epoch}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}')

    # Testing
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for batch in test_loader:
            # batch = batch.to(device)
            # z = model.encode(batch.x, batch.edge_index)
            # test_loss = model.recon_loss(z, batch.edge_index)
            batch = batch.to(device)
            x_hat, z = model(batch)
            loss = criterion(x_hat, batch.x)
            test_loss += loss.item()
        print(f'Test Loss: {test_loss / len(test_loader)}')

# def train(epoch):
#     model.train()
#     optimizer.zero_grad()
#     z = model.encode(x, train_pos_edge_index)
#     loss = model.recon_loss(z, train_pos_edge_index)
#     loss.backward()
#     optimizer.step()
    
#     writer.add_scalar("loss", loss.item(), epoch)

# def test(pos_edge_index, neg_edge_index):
#     model.eval()
#     with torch.no_grad():
#         z = model.encode(x, train_pos_edge_index)
#     return model.test(z, pos_edge_index, neg_edge_index)
    
class EmbeddedDataLoader(DataLoader):
    def __init__(self, *args, embedder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedder = embedder

    def __iter__(self):
        for batch in super().__iter__():
            # Apply the embedding function to the batch
            batch.x = self.embedder.embed_into_vector(batch.x)
            yield batch