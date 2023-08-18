import torch.nn as nn
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class Classifier_binary(nn.Module):
    """
    simple classifier, for prediction using latent space z
    https://dejanbatanjac.github.io/2019/07/04/softmax-vs-sigmoid.html
    """ 
    def __init__(self, dims):
        super(Classifier_binary, self).__init__()
        [in_dim, h_dims] = dims
        # assert out_dim == 1
        neurons = [in_dim, *h_dims] 
        linear_layers = [nn.Linear(
            neurons[i-1], neurons[i]) for i in range(1, len(neurons))]
        
        self.hidden = nn.ModuleList(linear_layers)
        self.final = nn.Linear(h_dims[-1], 1)

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
            # print(x.shape) [batch_size, h_dim[-1]]
        x = self.final(x)
        x = torch.sigmoid(x)
        return x

class fp_dataset(Dataset):
    def __init__(self, df):
        super(fp_dataset, self).__init__()
        self.len = len(df)
        self.df = df
    def __getitem__(self, idx):
        header = ['bit' + str(i) for i in range(167)]
        fp = self.df[header]
        fp = torch.tensor([float(b) for b in fp.iloc[idx]])
        label = self.df['Activity'][idx]
        # print(label)
        # label = onehot(2)(label)
        label = torch.tensor([label])
        return fp, label
    def __len__(self):
        return self.len
