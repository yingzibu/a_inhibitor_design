"""
Date: 8-17-2023

Adapted from paper: "Automatic Chemical Design using a data-driven
continuous representation of molecules"

pytorch adaptation reference:   
https://github.com/Ishan-Kumar2/Molecular_VAE_Pytorch/blob/master/model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# max_len = 120 # max_len of molecule
# alphabet_len = len(charset)

class MolecularVAE(nn.Module):
    def __init__(self, in_dim=[250, max_len, alphabet_len], z_dim=292):
        """
        param: in_dim = [batch_size, max_len, len(alphabet)]
        param: z_dim: z dimension
        """

        super(MolecularVAE, self).__init__()

        (_, max_len, alphabet_len) = in_dim
        self.max_len = max_len
        self.alphabet_len = alphabet_len
        self.kld = 0

        self.conv_1 = nn.Conv1d(max_len, 9, kernel_size = 9) #[bs, 9, 25]
        self.conv_2 = nn.Conv1d(9, 9, kernel_size=9) # [bs, 9, 17]
        self.conv_3 = nn.Conv1d(9, 10, kernel_size=11) # [bs, 10, 7]
        self.linear_0 = nn.Linear(70, 435)
        self.mu = nn.Linear(435, z_dim)
        self.logvar = nn.Linear(435, z_dim)

        self.linear_3 = nn.Linear(z_dim, z_dim)
        self.gru = nn.GRU(z_dim, 501, 3, batch_first=True)
        self.linear_4 = nn.Linear(501, alphabet_len)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()


    def encode(self, x):

        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        x = x.view(x.size(0), -1) # (batch_size, size(-2) * size(-1))
        # print('after view', x.shape)
        x = F.selu(self.linear_0(x))

        mu = self.mu(x)
        logvar = F.softplus(self.logvar(x))
        return mu, logvar

    def reparametrize(self, mu, logvar):
        epsilon = Variable(torch.rand(mu.size()), requires_grad=False)
        if mu.is_cuda: epsilon = epsilon.cuda()
        std = logvar.mul(0.5).exp_()
        z = mu.addcmul(std, epsilon)
        return z

    def decode(self, z):
        z = F.selu(self.linear_3(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, self.max_len, 1)
        output, hn = self.gru(z)
        out_reshape = output.contiguous().view(-1, output.size(-1))
        y0 = F.softmax(self.linear_4(out_reshape), dim=1)
        y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
        return y

    def update_kld(self, mu, logvar):
        self.kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def get_kld(self):
        return self.kld

    def forward(self, x):
        mu, logvar = self.encode(x)
        self.update_kld(mu, logvar)
        z = self.reparametrize(mu, logvar)
        return self.decode(z)

# from torchsummary import summary

# summary(MolecularVAE(), (250,120, 33))
# MolecularVAE()
