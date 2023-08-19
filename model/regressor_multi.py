"""
Date: 08-18-2023

Aim: To compare with MTATFP: https://github.com/Yimeng-Wang/JAK-MTATFP
     if simple NN regressor works, will use NN for property prediction 
     else will use MTATFP
     
Usage: check examples/fp_multi_label_regression.ipynb
"""

import torch.nn as nn
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class Regressor(nn.Module):
    """
    simple regression, for prediction using 
    latent space of dim [batch_size, z_dim]
    """ 
    def __init__(self, dims):
        """
        :param dims [in_dim, h_dims, out_dim]
            in_dim: input dim
            h_dims: a list of hidden dims
            out_dim: output dim
        """
        super(Regressor, self).__init__()
        [in_dim, h_dims, out_dim] = dims
        self.dims = dims
        neurons = [in_dim, *h_dims] 
        linear_layers = [nn.Linear(
            neurons[i-1], neurons[i]
            ) for i in range(1, len(neurons))]
        
        self.hidden = nn.ModuleList(linear_layers)
        self.final = nn.Linear(h_dims[-1], out_dim)

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        x = self.final(x)
        return x

class fp_reg_dataset(Dataset):
    """
    data set for loading data from MTATFP pIC50_enzyme
    using MACCS fingerprint as features
    """
    def __init__(self, df):
        super(fp_reg_dataset, self).__init__()
        self.len = len(df)
        self.df = df
    def __getitem__(self, idx):
        enzymes = ['JAK1', 'JAK2', 'JAK3', 'TYK2']
        header = ['bit' + str(i) for i in range(167)]
        fp = self.df[header]
        fp = torch.tensor([float(b) for b in fp.iloc[idx]], 
                          dtype=torch.float32)
        label = [self.df['pIC50_'+enzyme][idx] for enzyme in enzymes]
        label = torch.tensor(label, dtype=torch.float32) 
        return fp, label
    def __len__(self):
        return self.len
