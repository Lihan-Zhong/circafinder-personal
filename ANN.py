import pickle
import sys
import os
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from Plotting import *

class CircadianDataset(Dataset):
    def __init__(self, expr_matrix, times):
        """
        expr_matrix: numpy array of shape (N_cells, N_genes)
        times: numpy array of shape (N_cells,), use "hours" as unit, range: [0,24)
        """
        # Original expression matrix (float32), each row represents a cell and each column represents a gene
        self.X = expr_matrix.astype(np.float32)

        # Map the time from [0,24) to the Angle [0,2 π) on the circle and calculate sin/cos as the model label
        # This avoids the breakpoint problem of 23h ↔ 1h
        radians = times / 24.0 * 2 * np.pi
        self.y = np.stack([np.sin(radians), np.cos(radians)], axis=1).astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return (expression_vector, [sin(time), cos(time)])
        return self.X[idx], self.y[idx]
    

class CircadianNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout=0.2):
        super().__init__()
        layers = []

        # Feedforward Hidden Layer Design:
            # - First layer: Mapping from input_dim (number of genes) to 256 neurons
            # - Second Layer: 256 → 128
        # After each layer, ReLU activation is connected to enhance the nonlinear expression capability
        # And use Dropout(0.2) to prevent overfitting
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))  # Fully connected layer
            layers.append(nn.ReLU())               # Activation function
            layers.append(nn.Dropout(dropout))     # Random inactivation of 20%
            prev_dim = h

        # Output layer: Two neurons, corresponding respectively to sin(time) and cos(time)
        layers.append(nn.Linear(prev_dim, 2))

        # Combine all layers in sequence
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
    # Return mean loss
    return total_loss / len(dataloader.dataset)

def circular_mae(pred, true):
    """
    Calculate the Circular mean absolute error (Circular MAE) and the predicted sin/cos vector
    After converting to an Angle, calculate the minimum radian difference from the true Angle and then convert it to hours.
    """
    # atan2 param：sin, cos
    ang_pred = torch.atan2(pred[:,0], pred[:,1])
    ang_true = torch.atan2(true[:,0], true[:,1])
    # Normalize errors to [-π, π]
    diff = torch.remainder(ang_pred - ang_true + np.pi, 2*np.pi) - np.pi
    # Convert the radians to hours，24h correspondes to 2π
    return torch.mean(torch.abs(diff)) * (24/(2*np.pi))

def evaluate(model, dataloader, device):
    model.eval()
    total_mae = 0.0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            out = model(X_batch)
            total_mae += circular_mae(out, y_batch).item() * X_batch.size(0)
    return total_mae / len(dataloader.dataset)

def get_model_path_ann(cell_type):
    return "./Model_param/ANN/"+cell_type+'.pth'

def prediction_ann(adata,cell_types):
    predict_time_all = pd.Series(np.nan, index=adata.obs_names)
    predict_time_list = []

    for cell_type in cell_types:
        # print(cell_type)
        sub_adata = adata[adata.obs['cell_type'] == cell_type]
        expr_matrix = sub_adata.X.toarray()
        N_cells, N_genes = expr_matrix.shape
        model = CircadianNet(input_dim=N_genes)
        model_path = get_model_path_ann(cell_type)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        input_tensor = torch.from_numpy(expr_matrix)

        # Infer to obtain the predicted value of sin/cos
        with torch.no_grad():
            pred_vec = model(input_tensor)  # shape: (n_cells, 2)

        # Convert sin/cos to time (hours)
        pred_angles = torch.atan2(pred_vec[:, 0], pred_vec[:, 1])  # atan2(sin, cos)
        pred_hours = (pred_angles % (2 * np.pi)) * (24 / (2 * np.pi))

        sub_adata.obs['Predict time'] = pred_hours
        predict_time_list.append(pred_hours)

        predict_time_all.loc[sub_adata.obs_names] = pred_hours.numpy()
    
    fig = plot_histogram(predict_time_list, cell_types)
    # adata.obs['Predict time'] = predict_time_all
    adata.obs.loc[predict_time_all.notna(), 'Predict time'] = predict_time_all.dropna()
    # print(len(adata.obs))
    adata = adata[predict_time_all.notna()].copy()

    return fig, adata
    



