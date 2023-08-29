import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

class NoamOpt:
    "Optimizer wrapper that implements rate decay (adapted from\
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)"
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size

        self.state_dict = self.optimizer.state_dict()
        self.state_dict['step'] = 0
        self.state_dict['rate'] = 0

    def step(self):
        "Update parameters and rate"
        self.state_dict['step'] += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.state_dict['rate'] = rate
        self.optimizer.step()
        for k, v in self.optimizer.state_dict().items():
            self.state_dict[k] = v

    def rate(self, step=None):
        "Implement 'lrate' above"
        if step is None:
            step = self.state_dict['step']
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def load_state_dict(self, state_dict):
        self.state_dict = state_dict

class AdamOpt:
    "Adam optimizer wrapper"
    def __init__(self, params, lr, optimizer):
        self.optimizer = optimizer(params, lr)
        self.state_dict = self.optimizer.state_dict()

    def step(self):
        self.optimizer.step()
        self.state_dict = self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.state_dict = state_dict

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

import re
import math
import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan

rdBase.DisableLog('rdApp.*')


######## MODEL HELPERS ##########

def clones(module, N):
    """Produce N identical layers (adapted from
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    """Mask out subsequent positions (adapted from
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)"""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention' (adapted from Viswani et al.)"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class ListModule(nn.Module):
    """Create single pytorch module from list of modules"""
    def __init__(self, *args):
        super().__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

class KLAnnealer:
    """
    Scales KL weight (beta) linearly according to the number of epochs
    """
    def __init__(self, kl_low, kl_high, n_epochs, start_epoch):
        self.kl_low = kl_low
        self.kl_high = kl_high
        self.n_epochs = n_epochs
        self.start_epoch = start_epoch

        self.kl = (self.kl_high - self.kl_low) / (self.n_epochs - self.start_epoch)

    def __call__(self, epoch):
        k = (epoch - self.start_epoch) if epoch >= self.start_epoch else 0
        beta = self.kl_low + k * self.kl
        if beta > self.kl_high:
            beta = self.kl_high
        else:
            pass
        return beta


####### PREPROCESSING HELPERS ##########

def tokenizer(smile):
    "Tokenizes SMILES string"
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regezz = re.compile(pattern)
    tokens = [token for token in regezz.findall(smile)]
    assert smile == ''.join(tokens), ("{} could not be joined".format(smile))
    return tokens

def build_org_dict(char_dict):
    org_dict = {}
    for i, (k, v) in enumerate(char_dict.items()):
        if i == 0:
            pass
        else:
            org_dict[int(v-1)] = k
    return org_dict

def encode_smiles(smile, max_len, char_dict):
    "Converts tokenized SMILES string to list of token ids"
    for i in range(max_len - len(smile)):
        if i == 0:
            smile.append('<end>')
        else:
            smile.append('_')
    smile_vec = [char_dict[c] for c in smile]
    return smile_vec

def get_char_weights(train_smiles, params, freq_penalty=0.5):
    "Calculates token weights for a set of input data"
    char_dist = {}
    char_counts = np.zeros((params['NUM_CHAR'],))
    char_weights = np.zeros((params['NUM_CHAR'],))
    for k in params['CHAR_DICT'].keys():
        char_dist[k] = 0
    for smile in train_smiles:
        for i, char in enumerate(smile):
            char_dist[char] += 1
        for j in range(i, params['MAX_LENGTH']):
            char_dist['_'] += 1
    for i, v in enumerate(char_dist.values()):
        char_counts[i] = v
    top = np.sum(np.log(char_counts))
    for i in range(char_counts.shape[0]):
        char_weights[i] = top / np.log(char_counts[i])
    min_weight = char_weights.min()
    for i, w in enumerate(char_weights):
        if w > 2*min_weight:
            char_weights[i] = 2*min_weight
    scaler = MinMaxScaler([freq_penalty,1.0])
    char_weights = scaler.fit_transform(char_weights.reshape(-1, 1))
    return char_weights[:,0]


####### POSTPROCESSING HELPERS ##########

def decode_mols(encoded_tensors, org_dict):
    "Decodes tensor containing token ids into string"
    mols = []
    for i in range(encoded_tensors.shape[0]):
        encoded_tensor = encoded_tensors.cpu().numpy()[i,:] - 1
        mol_string = ''
        for i in range(encoded_tensor.shape[0]):
            idx = encoded_tensor[i]
            if org_dict[idx] == '<end>':
                break
            else:
                mol_string += org_dict[idx]
        mol_string = mol_string.strip('_')
        mols.append(mol_string)
    return mols

def calc_reconstruction_accuracies(input_smiles, output_smiles):
    "Calculates SMILE, token and positional accuracies for a set of\
    input and reconstructed SMILES strings"
    max_len = 126
    smile_accs = []
    hits = 0
    misses = 0
    position_accs = np.zeros((2, max_len))
    for in_smi, out_smi in zip(input_smiles, output_smiles):
        if in_smi == out_smi:
            smile_accs.append(1)
        else:
            smile_accs.append(0)

        misses += abs(len(in_smi) - len(out_smi))
        for j, (token_in, token_out) in enumerate(zip(in_smi, out_smi)):
            if token_in == token_out:
                hits += 1
                position_accs[0,j] += 1
            else:
                misses += 1
            position_accs[1,j] += 1

    smile_acc = np.mean(smile_accs)
    token_acc = hits / (hits + misses)
    position_acc = []
    for i in range(max_len):
        position_acc.append(position_accs[0,i] / position_accs[1,i])
    return smile_acc, token_acc, position_acc

def calc_entropy(sample):
    "Calculates Shannon information entropy for a set of input memories"
    es = []
    for i in range(sample.shape[1]):
        probs, bin_edges = np.histogram(sample[:,i], bins=1000, range=(-5., 5.), density=True)
        es.append(entropy(probs))
    return np.array(es)

####### ADDITIONAL METRIC CALCULATIONS #########

def load_gen(path):
    "Loads set of generated SMILES strings from path"
    smiles = pd.read_csv(path).SMILES.to_list()
    return smiles

def valid(smiles):
    "Returns valid SMILES (RDKit sanitizable) from a set of\
    SMILES strings"
    valid_smiles = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            pass
        else:
            try:
                Chem.SanitizeMol(mol)
                valid_smiles.append(smi)
            except ValueError:
                pass
    return valid_smiles

def calc_token_lengths(smiles):
    "Calculates the token lengths of a set of SMILES strings"
    lens = []
    for smi in smiles:
        smi = tokenizer(smi)
        lens.append(len(smi))
    return lens

def calc_MW(smiles):
    "Calculates the molecular weights of a set of SMILES strings"
    MWs = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        MWs.append(Descriptors.MolWt(mol))
    return MWs

def novel(smiles, train_smiles):
    "Returns novel SMILES strings that do not appear\
    in training set"
    set_smiles = set(smiles)
    set_train = set(train_smiles)
    novel_smiles = list(set_smiles - set_train)
    return novel_smiles

def unique(smiles):
    "Returns unique SMILES strings from set"
    unique_smiles = set(smiles)
    return list(unique_smiles)

def fingerprints(smiles):
    "Calculates fingerprints of a list of SMILES strings"
    fps = np.zeros((len(smiles), 1024))
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        fp = np.asarray(Morgan(mol, 2, 1024), dtype='uint8')
        fps[i,:] = fp
    return fps

def tanimoto_similarity(bv1, bv2):
    "Calculates Tanimoto similarity between two fingerprint bit vectors"
    mand = sum(bv1 & bv2)
    mor = sum(bv1 | bv2)
    return mand / mor

def pass_through_filters(smiles, data_dir='data'):
    """Filters SMILES strings based on method implemented in
    http://nlp.seas.harvard.edu/2018/04/03/attention.html"""
    _mcf = pd.read_csv('{}/mcf.csv'.format(data_dir))
    _pains = pd.read_csv('{}/wehi_pains.csv'.format(data_dir), names=['smarts', 'names'])
    _filters = [Chem.MolFromSmarts(x) for x in
                _mcf.append(_pains, sort=True)['smarts'].values]
    filtered_smiles = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        h_mol = Chem.AddHs(mol)
        filtered = False
        if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
            filtered = True
        if any(h_mol.HasSubstructMatch(smarts) for smarts in _filters):
            filtered = True
        if not filtered:
            filtered_smiles.append(smi)
    return filtered_smiles

def cross_diversity(set1, set2, bs1=5000, bs2=5000, p=1, agg='max',
                    device='cpu'):
    """
    Function for calculating the maximum average tanimoto similarity score
    between the generated set and the training set (this code is adapted from
    https://github.com/molecularsets/moses)
    """
    agg_tanimoto = np.zeros(len(set2))
    total = np.zeros(len(set2))
    set2 = torch.tensor(set2).to(device).float()
    for j in range(0, set1.shape[0], bs1):
        x_stock = torch.tensor(set1[j:j+bs1]).to(device).float()
        for i in range(0, set2.shape[0], bs2):
            y_gen = set2[i:i+bs2]
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                   y_gen.sum(0, keepdim=True) -tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p!= 1:
                jac = jac**p
            if agg == 'max':
                agg_tanimoto[i:i+y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i+y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i+y_gen.shape[1]] += jac.sum(0)
                total[i:i+y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1/p)
    return 1 - np.mean(agg_tanimoto)


####### GRADIENT TROUBLESHOOTING #########

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    layers = np.array(layers)
    ave_grads = np.array(ave_grads)
    fig = plt.figure(figsize=(12,6))
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.ylim(ymin=0, ymax=5e-3)
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    return plt



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

def vae_loss(x, x_out, mu, logvar, true_prop, pred_prop, weights, beta=1):
    "Binary Cross Entropy Loss + Kiebler-Lublach Divergence"
    x = x.long()[:,1:] - 1
    x = x.contiguous().view(-1)
    x_out = x_out.contiguous().view(-1, x_out.size(2))
    BCE = F.cross_entropy(x_out, x, reduction='mean', weight=weights)
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if pred_prop is not None:
        MSE = F.mse_loss(pred_prop.squeeze(-1), true_prop)
    else:
        MSE = torch.tensor(0.)
    if torch.isnan(KLD):
        KLD = torch.tensor(0.)
    return BCE + KLD + MSE, BCE, KLD, MSE

def trans_vae_loss(x, x_out, mu, logvar, true_len, pred_len, true_prop, pred_prop, weights, beta=1):
    "Binary Cross Entropy Loss + Kiebler-Lublach Divergence + Mask Length Prediction"
    x = x.long()[:,1:] - 1
    x = x.contiguous().view(-1)
    x_out = x_out.contiguous().view(-1, x_out.size(2))
    true_len = true_len.contiguous().view(-1)
    BCEmol = F.cross_entropy(x_out, x, reduction='mean', weight=weights)
    BCEmask = F.cross_entropy(pred_len, true_len, reduction='mean')
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if pred_prop is not None:
        MSE = F.mse_loss(pred_prop.squeeze(-1), true_prop)
    else:
        MSE = torch.tensor(0.)
    if torch.isnan(KLD):
        KLD = torch.tensor(0.)
    return BCEmol + BCEmask + KLD + MSE, BCEmol, BCEmask, KLD, MSE


import os
import json
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

# from transvae.tvae_util import *
# from transvae.opt import NoamOpt
# from transvae.data import vae_data_gen, make_std_mask
# from transvae.loss import vae_loss, trans_vae_loss


####### MODEL SHELL ##########

class VAEShell():
    """
    VAE shell class that includes methods for parameter initiation,
    data loading, training, logging, checkpointing, loading and saving,
    """
    def __init__(self, params, name=None):
        self.params = params
        self.name = name
        if 'BATCH_SIZE' not in self.params.keys():
            self.params['BATCH_SIZE'] = 500
        if 'BATCH_CHUNKS' not in self.params.keys():
            self.params['BATCH_CHUNKS'] = 5
        if 'BETA_INIT' not in self.params.keys():
            self.params['BETA_INIT'] = 1e-8
        if 'BETA' not in self.params.keys():
            self.params['BETA'] = 0.05
        if 'ANNEAL_START' not in self.params.keys():
            self.params['ANNEAL_START'] = 0
        if 'LR' not in self.params.keys():
            self.params['LR_SCALE'] = 1
        if 'WARMUP_STEPS' not in self.params.keys():
            self.params['WARMUP_STEPS'] = 10000
        if 'EPS_SCALE' not in self.params.keys():
            self.params['EPS_SCALE'] = 1
        if 'CHAR_DICT' in self.params.keys():
            self.vocab_size = len(self.params['CHAR_DICT'].keys())
            self.pad_idx = self.params['CHAR_DICT']['_']
            if 'CHAR_WEIGHTS' in self.params.keys():
                self.params['CHAR_WEIGHTS'] = torch.tensor(self.params['CHAR_WEIGHTS'], dtype=torch.float)
            else:
                self.params['CHAR_WEIGHTS'] = torch.ones(self.vocab_size, dtype=torch.float)
        self.loss_func = vae_loss
        self.data_gen = vae_data_gen

        ### Sequence length hard-coded into model
        self.src_len = 126
        self.tgt_len = 125

        ### Build empty structures for data storage
        self.n_epochs = 0
        self.best_loss = np.inf
        self.current_state = {'name': self.name,
                              'epoch': self.n_epochs,
                              'model_state_dict': None,
                              'optimizer_state_dict': None,
                              'best_loss': self.best_loss,
                              'params': self.params}
        self.loaded_from = None

    def save(self, state, fn, path='checkpoints', use_name=True):
        """
        Saves current model state to .ckpt file

        Arguments:
            state (dict, required): Dictionary containing model state
            fn (str, required): File name to save checkpoint with
            path (str): Folder to store saved checkpoints
        """
        os.makedirs(path, exist_ok=True)
        if use_name:
            if os.path.splitext(fn)[1] == '':
                if self.name is not None:
                    fn += '_' + self.name
                fn += '.ckpt'
            else:
                if self.name is not None:
                    fn, ext = fn.split('.')
                    fn += '_' + self.name
                    fn += '.' + ext
            save_path = os.path.join(path, fn)
        else:
            save_path = fn
        torch.save(state, save_path)

    def load(self, checkpoint_path):
        """
        Loads a saved model state

        Arguments:
            checkpoint_path (str, required): Path to saved .ckpt file
        """
        loaded_checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.loaded_from = checkpoint_path
        for k in self.current_state.keys():
            try:
                self.current_state[k] = loaded_checkpoint[k]
            except KeyError:
                self.current_state[k] = None

        if self.name is None:
            self.name = self.current_state['name']
        else:
            pass
        self.n_epochs = self.current_state['epoch']
        self.best_loss = self.current_state['best_loss']
        for k, v in self.current_state['params'].items():
            if k in self.arch_params or k not in self.params.keys():
                self.params[k] = v
            else:
                pass
        self.vocab_size = len(self.params['CHAR_DICT'].keys())
        self.pad_idx = self.params['CHAR_DICT']['_']
        self.build_model()
        self.model.load_state_dict(self.current_state['model_state_dict'])
        self.optimizer.load_state_dict(self.current_state['optimizer_state_dict'])

    def train(self, train_mols, val_mols, train_props=None, val_props=None,
              epochs=100, save=True, save_freq=None, log=True, log_dir='trials'):
        """
        Train model and validate

        Arguments:
            train_mols (np.array, required): Numpy array containing training
                                             molecular structures
            val_mols (np.array, required): Same format as train_mols. Used for
                                           model development or validation
            train_props (np.array): Numpy array containing chemical property of
                                   molecular structure
            val_props (np.array): Same format as train_prop. Used for model
                                 development or validation
            epochs (int): Number of epochs to train the model for
            save (bool): If true, saves latest and best versions of model
            save_freq (int): Frequency with which to save model checkpoints
            log (bool): If true, writes training metrics to log file
            log_dir (str): Directory to store log files
        """
        ### Prepare data iterators
        train_data = self.data_gen(train_mols, train_props, char_dict=self.params['CHAR_DICT'])
        val_data = self.data_gen(val_mols, val_props, char_dict=self.params['CHAR_DICT'])

        train_iter = torch.utils.data.DataLoader(train_data,
                                                 batch_size=self.params['BATCH_SIZE'],
                                                 shuffle=True, num_workers=0,
                                                 pin_memory=False, drop_last=True)
        val_iter = torch.utils.data.DataLoader(val_data,
                                               batch_size=self.params['BATCH_SIZE'],
                                               shuffle=True, num_workers=0,
                                               pin_memory=False, drop_last=True)
        self.chunk_size = self.params['BATCH_SIZE'] // self.params['BATCH_CHUNKS']


        torch.backends.cudnn.benchmark = True

        ### Determine save frequency
        if save_freq is None:
            save_freq = epochs

        ### Setup log file
        if log:
            os.makedirs(log_dir, exist_ok=True)
            if self.name is not None:
                log_fn = '{}/log{}.txt'.format(log_dir, '_'+self.name)
            else:
                log_fn = '{}/log.txt'.format(log_dir)
            try:
                f = open(log_fn, 'r')
                f.close()
                already_wrote = True
            except FileNotFoundError:
                already_wrote = False
            log_file = open(log_fn, 'a')
            if not already_wrote:
                log_file.write('epoch,batch_idx,data_type,tot_loss,recon_loss,pred_loss,kld_loss,prop_mse_loss,run_time\n')
            log_file.close()

        ### Initialize Annealer
        kl_annealer = KLAnnealer(self.params['BETA_INIT'], self.params['BETA'],
                                 epochs, self.params['ANNEAL_START'])

        ### Epoch loop
        for epoch in range(epochs):
            ### Train Loop
            self.model.train()
            losses = []
            beta = kl_annealer(epoch)
            for j, data in enumerate(train_iter):
                avg_losses = []
                avg_bce_losses = []
                avg_bcemask_losses = []
                avg_kld_losses = []
                avg_prop_mse_losses = []
                start_run_time = perf_counter()
                for i in range(self.params['BATCH_CHUNKS']):
                    batch_data = data[i*self.chunk_size:(i+1)*self.chunk_size,:]
                    mols_data = batch_data[:,:-1]
                    props_data = batch_data[:,-1]
                    if self.use_gpu:
                        mols_data = mols_data.cuda()
                        props_data = props_data.cuda()


                    src = Variable(mols_data).long()
                    tgt = Variable(mols_data[:,:-1]).long()
                    true_prop = Variable(props_data)
                    src_mask = (src != self.pad_idx).unsqueeze(-2)
                    tgt_mask = make_std_mask(tgt, self.pad_idx)

                    if self.model_type == 'transformer':
                        x_out, mu, logvar, pred_len, pred_prop = self.model(src, tgt, src_mask, tgt_mask)
                        true_len = src_mask.sum(dim=-1)
                        loss, bce, bce_mask, kld, prop_mse = trans_vae_loss(src, x_out, mu, logvar,
                                                                            true_len, pred_len,
                                                                            true_prop, pred_prop,
                                                                            self.params['CHAR_WEIGHTS'],
                                                                            beta)
                        avg_bcemask_losses.append(bce_mask.item())
                    else:
                        x_out, mu, logvar, pred_prop = self.model(src, tgt, src_mask, tgt_mask)
                        loss, bce, kld, prop_mse = self.loss_func(src, x_out, mu, logvar,
                                                                  true_prop, pred_prop,
                                                                  self.params['CHAR_WEIGHTS'],
                                                                  beta)
                    avg_losses.append(loss.item())
                    avg_bce_losses.append(bce.item())
                    avg_kld_losses.append(kld.item())
                    avg_prop_mse_losses.append(prop_mse.item())
                    loss.backward()
                self.optimizer.step()
                self.model.zero_grad()
                stop_run_time = perf_counter()
                run_time = round(stop_run_time - start_run_time, 5)
                avg_loss = np.mean(avg_losses)
                avg_bce = np.mean(avg_bce_losses)
                if len(avg_bcemask_losses) == 0:
                    avg_bcemask = 0
                else:
                    avg_bcemask = np.mean(avg_bcemask_losses)
                avg_kld = np.mean(avg_kld_losses)
                avg_prop_mse = np.mean(avg_prop_mse_losses)
                losses.append(avg_loss)

                if log:
                    log_file = open(log_fn, 'a')
                    log_file.write('{},{},{},{},{},{},{},{},{}\n'.format(self.n_epochs,
                                                                         j, 'train',
                                                                         avg_loss,
                                                                         avg_bce,
                                                                         avg_bcemask,
                                                                         avg_kld,
                                                                         avg_prop_mse,
                                                                         run_time))
                    log_file.close()
            train_loss = np.mean(losses)

            ### Val Loop
            self.model.eval()
            losses = []
            for j, data in enumerate(val_iter):
                avg_losses = []
                avg_bce_losses = []
                avg_bcemask_losses = []
                avg_kld_losses = []
                avg_prop_mse_losses = []
                start_run_time = perf_counter()
                for i in range(self.params['BATCH_CHUNKS']):
                    batch_data = data[i*self.chunk_size:(i+1)*self.chunk_size,:]
                    mols_data = batch_data[:,:-1]
                    props_data = batch_data[:,-1]
                    if self.use_gpu:
                        mols_data = mols_data.cuda()
                        props_data = props_data.cuda()

                    src = Variable(mols_data).long()
                    tgt = Variable(mols_data[:,:-1]).long()
                    true_prop = Variable(props_data)
                    src_mask = (src != self.pad_idx).unsqueeze(-2)
                    tgt_mask = make_std_mask(tgt, self.pad_idx)
                    scores = Variable(data[:,-1])

                    if self.model_type == 'transformer':
                        x_out, mu, logvar, pred_len, pred_prop = self.model(src, tgt, src_mask, tgt_mask)
                        true_len = src_mask.sum(dim=-1)
                        loss, bce, bce_mask, kld, prop_mse = trans_vae_loss(src, x_out, mu, logvar,
                                                                            true_len, pred_len,
                                                                            true_prop, pred_prop,
                                                                            self.params['CHAR_WEIGHTS'],
                                                                            beta)
                        avg_bcemask_losses.append(bce_mask.item())
                    else:
                        x_out, mu, logvar, pred_prop = self.model(src, tgt, src_mask, tgt_mask)
                        loss, bce, kld, prop_mse = self.loss_func(src, x_out, mu, logvar,
                                                                  true_prop, pred_prop,
                                                                  self.params['CHAR_WEIGHTS'],
                                                                  beta)
                    avg_losses.append(loss.item())
                    avg_bce_losses.append(bce.item())
                    avg_kld_losses.append(kld.item())
                    avg_prop_mse_losses.append(prop_mse.item())
                stop_run_time = perf_counter()
                run_time = round(stop_run_time - start_run_time, 5)
                avg_loss = np.mean(avg_losses)
                avg_bce = np.mean(avg_bce_losses)
                if len(avg_bcemask_losses) == 0:
                    avg_bcemask = 0
                else:
                    avg_bcemask = np.mean(avg_bcemask_losses)
                avg_kld = np.mean(avg_kld_losses)
                avg_prop_mse = np.mean(avg_prop_mse_losses)
                losses.append(avg_loss)

                if log:
                    log_file = open(log_fn, 'a')
                    log_file.write('{},{},{},{},{},{},{},{}\n'.format(self.n_epochs,
                                                                j, 'test',
                                                                avg_loss,
                                                                avg_bce,
                                                                avg_bcemask,
                                                                avg_kld,
                                                                avg_prop_mse,
                                                                run_time))
                    log_file.close()

            self.n_epochs += 1
            val_loss = np.mean(losses)
            print('Epoch - {} Train - {} Val - {} KLBeta - {}'.format(self.n_epochs, train_loss, val_loss, beta))

            ### Update current state and save model
            self.current_state['epoch'] = self.n_epochs
            self.current_state['model_state_dict'] = self.model.state_dict()
            self.current_state['optimizer_state_dict'] = self.optimizer.state_dict

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.current_state['best_loss'] = self.best_loss
                if save:
                    self.save(self.current_state, 'best')

            if (self.n_epochs) % save_freq == 0:
                epoch_str = str(self.n_epochs)
                while len(epoch_str) < 3:
                    epoch_str = '0' + epoch_str
                if save:
                    self.save(self.current_state, epoch_str)

    ### Sampling and Decoding Functions
    def sample_from_memory(self, size, mode='rand', sample_dims=None, k=5):
        """
        Quickly sample from latent dimension

        Arguments:
            size (int, req): Number of samples to generate in one batch
            mode (str): Sampling mode (rand, high_entropy or k_high_entropy)
            sample_dims (list): List of dimensions to sample from if mode is
                                high_entropy or k_high_entropy
            k (int): Number of high entropy dimensions to randomly sample from
        Returns:
            z (torch.tensor): NxD_latent tensor containing sampled memory vectors
        """
        if mode == 'rand':
            z = torch.randn(size, self.params['d_latent'])
        else:
            assert sample_dims is not None, "ERROR: Must provide sample dimensions"
            if mode == 'top_dims':
                z = torch.zeros((size, self.params['d_latent']))
                for d in sample_dims:
                    z[:,d] = torch.randn(size)
            elif mode == 'k_dims':
                z = torch.zeros((size, self.params['d_latent']))
                d_select = np.random.choice(sample_dims, size=k, replace=False)
                for d in d_select:
                    z[:,d] = torch.randn(size)
        return z

    def greedy_decode(self, mem, src_mask=None, condition=[]):
        """
        Greedy decode from model memory

        Arguments:
            mem (torch.tensor, req): Memory tensor to send to decoder
            src_mask (torch.tensor): Mask tensor to hide padding tokens (if
                                     model_type == 'transformer')
        Returns:
            decoded (torch.tensor): Tensor of predicted token ids
        """
        start_symbol = self.params['CHAR_DICT']['<start>']
        max_len = self.tgt_len
        decoded = torch.ones(mem.shape[0],1).fill_(start_symbol).long()
        for tok in condition:
            condition_symbol = self.params['CHAR_DICT'][tok]
            condition_vec = torch.ones(mem.shape[0],1).fill_(condition_symbol).long()
            decoded = torch.cat([decoded, condition_vec], dim=1)
        tgt = torch.ones(mem.shape[0],max_len+1).fill_(start_symbol).long()
        tgt[:,:len(condition)+1] = decoded
        if src_mask is None and self.model_type == 'transformer':
            mask_lens = self.model.encoder.predict_mask_length(mem)
            src_mask = torch.zeros((mem.shape[0], 1, self.src_len+1))
            for i in range(mask_lens.shape[0]):
                mask_len = mask_lens[i].item()
                src_mask[i,:,:mask_len] = torch.ones((1, 1, mask_len))
        elif self.model_type != 'transformer':
            src_mask = torch.ones((mem.shape[0], 1, self.src_len))

        if self.use_gpu:
            src_mask = src_mask.cuda()
            decoded = decoded.cuda()
            tgt = tgt.cuda()

        self.model.eval()
        for i in range(len(condition), max_len):
            if self.model_type == 'transformer':
                decode_mask = Variable(subsequent_mask(decoded.size(1)).long())
                if self.use_gpu:
                    decode_mask = decode_mask.cuda()
                out = self.model.decode(mem, src_mask, Variable(decoded),
                                        decode_mask)
            else:
                out, _ = self.model.decode(tgt, mem)
            out = self.model.generator(out)
            prob = F.softmax(out[:,i,:], dim=-1)
            _, next_word = torch.max(prob, dim=1)
            next_word += 1
            tgt[:,i+1] = next_word
            if self.model_type == 'transformer':
                next_word = next_word.unsqueeze(1)
                decoded = torch.cat([decoded, next_word], dim=1)
        decoded = tgt[:,1:]
        return decoded

    def reconstruct(self, data, method='greedy', log=True, return_mems=True, return_str=True):
        """
        Method for encoding input smiles into memory and decoding back
        into smiles

        Arguments:
            data (np.array, required): Input array consisting of smiles and property
            method (str): Method for decoding. Greedy decoding is currently the only
                          method implemented. May implement beam search, top_p or top_k
                          in future versions.
            log (bool): If true, tracks reconstruction progress in separate log file
            return_mems (bool): If true, returns memory vectors in addition to decoded SMILES
            return_str (bool): If true, translates decoded vectors into SMILES strings. If false
                               returns tensor of token ids
        Returns:
            decoded_smiles (list): Decoded smiles data - either decoded SMILES strings or tensor of
                                   token ids
            mems (np.array): Array of model memory vectors
        """
        data = vae_data_gen(data, props=None, char_dict=self.params['CHAR_DICT'])

        data_iter = torch.utils.data.DataLoader(data,
                                                batch_size=self.params['BATCH_SIZE'],
                                                shuffle=False, num_workers=0,
                                                pin_memory=False, drop_last=True)
        self.batch_size = self.params['BATCH_SIZE']
        self.chunk_size = self.batch_size // self.params['BATCH_CHUNKS']

        self.model.eval()
        decoded_smiles = []
        mems = torch.empty((data.shape[0], self.params['d_latent'])).cpu()
        for j, data in enumerate(data_iter):
            if log:
                log_file = open('calcs/{}_progress.txt'.format(self.name), 'a')
                log_file.write('{}\n'.format(j))
                log_file.close()
            for i in range(self.params['BATCH_CHUNKS']):
                batch_data = data[i*self.chunk_size:(i+1)*self.chunk_size,:]
                mols_data = batch_data[:,:-1]
                props_data = batch_data[:,-1]
                if self.use_gpu:
                    mols_data = mols_data.cuda()
                    props_data = props_data.cuda()

                src = Variable(mols_data).long()
                src_mask = (src != self.pad_idx).unsqueeze(-2)

                ### Run through encoder to get memory
                if self.model_type == 'transformer':
                    _, mem, _, _ = self.model.encode(src, src_mask)
                else:
                    _, mem, _ = self.model.encode(src)
                start = j*self.batch_size+i*self.chunk_size
                stop = j*self.batch_size+(i+1)*self.chunk_size
                mems[start:stop, :] = mem.detach().cpu()

                ### Decode logic
                if method == 'greedy':
                    decoded = self.greedy_decode(mem, src_mask=src_mask)
                else:
                    decoded = None

                if return_str:
                    decoded = decode_mols(decoded, self.params['ORG_DICT'])
                    decoded_smiles += decoded
                else:
                    decoded_smiles.append(decoded)

        if return_mems:
            return decoded_smiles, mems.detach().numpy()
        else:
            return decoded_smiles

    def sample(self, n, method='greedy', sample_mode='rand',
                        sample_dims=None, k=None, return_str=True,
                        condition=[]):
        """
        Method for sampling from memory and decoding back into SMILES strings

        Arguments:
            n (int): Number of data points to sample
            method (str): Method for decoding. Greedy decoding is currently the only
                          method implemented. May implement beam search, top_p or top_k
                          in future versions.
            sample_mode (str): Sampling mode (rand, high_entropy or k_high_entropy)
            sample_dims (list): List of dimensions to sample from if mode is
                                high_entropy or k_high_entropy
            k (int): Number of high entropy dimensions to randomly sample from
            return_str (bool): If true, translates decoded vectors into SMILES strings. If false
                               returns tensor of token ids
        Returns:
            decoded (list): Decoded smiles data - either decoded SMILES strings or tensor of
                            token ids
        """
        mem = self.sample_from_memory(n, mode=sample_mode, sample_dims=sample_dims, k=k)

        if self.use_gpu:
            mem = mem.cuda()

        ### Decode logic
        if method == 'greedy':
            decoded = self.greedy_decode(mem, condition=condition)
        else:
            decoded = None

        if return_str:
            decoded = decode_mols(decoded, self.params['ORG_DICT'])
        return decoded

    def calc_mems(self, data, log=True, save_dir='memory', save_fn='model_name', save=True):
        """
        Method for calculating and saving the memory of each neural net

        Arguments:
            data (np.array, req): Input array containing SMILES strings
            log (bool): If true, tracks calculation progress in separate log file
            save_dir (str): Directory to store output memory array
            save_fn (str): File name to store output memory array
            save (bool): If true, saves memory to disk. If false, returns memory
        Returns:
            mems(np.array): Reparameterized memory array
            mus(np.array): Mean memory array (prior to reparameterization)
            logvars(np.array): Log variance array (prior to reparameterization)
        """
        data = vae_data_gen(data, props=None, char_dict=self.params['CHAR_DICT'])

        data_iter = torch.utils.data.DataLoader(data,
                                                batch_size=self.params['BATCH_SIZE'],
                                                shuffle=False, num_workers=0,
                                                pin_memory=False, drop_last=True)
        save_shape = len(data_iter)*self.params['BATCH_SIZE']
        self.batch_size = self.params['BATCH_SIZE']
        self.chunk_size = self.batch_size // self.params['BATCH_CHUNKS']
        mems = torch.empty((save_shape, self.params['d_latent'])).cpu()
        mus = torch.empty((save_shape, self.params['d_latent'])).cpu()
        logvars = torch.empty((save_shape, self.params['d_latent'])).cpu()

        self.model.eval()
        for j, data in enumerate(data_iter):
            if log:
                log_file = open('memory/{}_progress.txt'.format(self.name), 'a')
                log_file.write('{}\n'.format(j))
                log_file.close()
            for i in range(self.params['BATCH_CHUNKS']):
                batch_data = data[i*self.chunk_size:(i+1)*self.chunk_size,:]
                mols_data = batch_data[:,:-1]
                props_data = batch_data[:,-1]
                if self.use_gpu:
                    mols_data = mols_data.cuda()
                    props_data = props_data.cuda()

                src = Variable(mols_data).long()
                src_mask = (src != self.pad_idx).unsqueeze(-2)

                ### Run through encoder to get memory
                if self.model_type == 'transformer':
                    mem, mu, logvar, _ = self.model.encode(src, src_mask)
                else:
                    mem, mu, logvar = self.model.encode(src)
                start = j*self.batch_size+i*self.chunk_size
                stop = j*self.batch_size+(i+1)*self.chunk_size
                mems[start:stop, :] = mem.detach().cpu()
                mus[start:stop, :] = mu.detach().cpu()
                logvars[start:stop, :] = logvar.detach().cpu()

        if save:
            if save_fn == 'model_name':
                save_fn = self.name
            save_path = os.path.join(save_dir, save_fn)
            np.save('{}_mems.npy'.format(save_path), mems.detach().numpy())
            np.save('{}_mus.npy'.format(save_path), mus.detach().numpy())
            np.save('{}_logvars.npy'.format(save_path), logvars.detach().numpy())
        else:
            return mems.detach().numpy(), mus.detach().numpy(), logvars.detach().numpy()


####### Encoder, Decoder and Generator ############

class TransVAE(VAEShell):
    """
    Transformer-based VAE class. Between the encoder and decoder is a stochastic
    latent space. "Memory value" matrices are convolved to latent bottleneck and
    deconvolved before being sent to source attention in decoder.
    """
    def __init__(self, params={}, name=None, N=3, d_model=128, d_ff=512,
                 d_latent=128, h=4, dropout=0.1, bypass_bottleneck=False,
                 property_predictor=False, d_pp=256, depth_pp=2, load_fn=None):
        super().__init__(params, name)
        """
        Instatiating a TransVAE object builds the model architecture, data structs
        to store the model parameters and training information and initiates model
        weights. Most params have default options but vocabulary must be provided.

        Arguments:
            params (dict, required): Dictionary with model parameters. Keys must match
                                     those written in this module
            name (str): Name of model (all save and log files will be written with
                        this name)
            N (int): Number of repeat encoder and decoder layers
            d_model (int): Dimensionality of model (embeddings and attention)
            d_ff (int): Dimensionality of feed-forward layers
            d_latent (int): Dimensionality of latent space
            h (int): Number of heads per attention layer
            dropout (float): Rate of dropout
            bypass_bottleneck (bool): If false, model functions as standard autoencoder
            property_predictor (bool): If true, model will predict property from latent memory
            d_pp (int): Dimensionality of property predictor layers
            depth_pp (int): Number of property predictor layers
            load_fn (str): Path to checkpoint file
        """

        ### Store architecture params
        self.model_type = 'transformer'
        self.params['model_type'] = self.model_type
        self.params['N'] = N
        self.params['d_model'] = d_model
        self.params['d_ff'] = d_ff
        self.params['d_latent'] = d_latent
        self.params['h'] = h
        self.params['dropout'] = dropout
        self.params['bypass_bottleneck'] = bypass_bottleneck
        self.params['property_predictor'] = property_predictor
        self.params['d_pp'] = d_pp
        self.params['depth_pp'] = depth_pp
        self.arch_params = ['N', 'd_model', 'd_ff', 'd_latent', 'h', 'dropout', 'bypass_bottleneck',
                            'property_predictor', 'd_pp', 'depth_pp']

        ### Build model architecture
        if load_fn is None:
            self.build_model()
        else:
            self.load(load_fn)

    def build_model(self):
        """
        Build model architecture. This function is called during initialization as well as when
        loading a saved model checkpoint
        """
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.params['h'], self.params['d_model'])
        ff = PositionwiseFeedForward(self.params['d_model'], self.params['d_ff'], self.params['dropout'])
        position = PositionalEncoding(self.params['d_model'], self.params['dropout'])
        encoder = VAEEncoder(EncoderLayer(self.params['d_model'], self.src_len, c(attn), c(ff), self.params['dropout']),
                                          self.params['N'], self.params['d_latent'], self.params['bypass_bottleneck'],
                                          self.params['EPS_SCALE'])
        decoder = VAEDecoder(EncoderLayer(self.params['d_model'], self.src_len, c(attn), c(ff), self.params['dropout']),
                             DecoderLayer(self.params['d_model'], self.tgt_len, c(attn), c(attn), c(ff), self.params['dropout']),
                                          self.params['N'], self.params['d_latent'], self.params['bypass_bottleneck'])
        src_embed = nn.Sequential(Embeddings(self.params['d_model'], self.vocab_size), c(position))
        tgt_embed = nn.Sequential(Embeddings(self.params['d_model'], self.vocab_size), c(position))
        generator = Generator(self.params['d_model'], self.vocab_size)
        if self.params['property_predictor']:
            property_predictor = PropertyPredictor(self.params['d_pp'], self.params['depth_pp'], self.params['d_latent'])
        else:
            property_predictor = None
        self.model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator, property_predictor)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.model.cuda()
            self.params['CHAR_WEIGHTS'] = self.params['CHAR_WEIGHTS'].cuda()

        ### Initiate optimizer
        self.optimizer = NoamOpt(self.params['d_model'], self.params['LR_SCALE'], self.params['WARMUP_STEPS'],
                                 torch.optim.Adam(self.model.parameters(), lr=0,
                                 betas=(0.9,0.98), eps=1e-9))

class EncoderDecoder(nn.Module):
    """
    Base transformer Encoder-Decoder architecture
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, property_predictor):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.property_predictor = property_predictor

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and tgt sequences"
        mem, mu, logvar, pred_len = self.encode(src, src_mask)
        x = self.decode(mem, src_mask, tgt, tgt_mask)
        x = self.generator(x)
        if self.property_predictor is not None:
            prop = self.predict_property(mu)
        else:
            prop = None
        return x, mu, logvar, pred_len, prop

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, mem, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), mem, src_mask, tgt_mask)

    def predict_property(self, mu):
        return self.property_predictor(mu)

class Generator(nn.Module):
    "Generates token predictions after final decoder layer"
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab-1)

    def forward(self, x):
        return self.proj(x)

class VAEEncoder(nn.Module):
    "Base transformer encoder architecture"
    def __init__(self, layer, N, d_latent, bypass_bottleneck, eps_scale):
        super().__init__()
        self.layers = clones(layer, N)
        self.conv_bottleneck = ConvBottleneck(layer.size)
        self.z_means, self.z_var = nn.Linear(576, d_latent), nn.Linear(576, d_latent)
        self.norm = LayerNorm(layer.size)
        self.predict_len1 = nn.Linear(d_latent, d_latent*2)
        self.predict_len2 = nn.Linear(d_latent*2, d_latent)

        self.bypass_bottleneck = bypass_bottleneck
        self.eps_scale = eps_scale

    def predict_mask_length(self, mem):
        "Predicts mask length from latent memory so mask can be re-created during inference"
        pred_len = self.predict_len1(mem)
        pred_len = self.predict_len2(pred_len)
        pred_len = F.softmax(pred_len, dim=-1)
        pred_len = torch.topk(pred_len, 1)[1]
        return pred_len

    def reparameterize(self, mu, logvar, eps_scale=1):
        "Stochastic reparameterization"
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) * eps_scale
        return mu + eps*std

    def forward(self, x, mask):
        ### Attention and feedforward layers
        for i, attn_layer in enumerate(self.layers):
            x = attn_layer(x, mask)
        ### Batch normalization
        mem = self.norm(x)
        ### Convolutional Bottleneck
        if self.bypass_bottleneck:
            mu, logvar = Variable(torch.tensor([0.0])), Variable(torch.tensor([0.0]))
        else:
            mem = mem.permute(0, 2, 1)
            mem = self.conv_bottleneck(mem)
            mem = mem.contiguous().view(mem.size(0), -1)
            mu, logvar = self.z_means(mem), self.z_var(mem)
            mem = self.reparameterize(mu, logvar, self.eps_scale)
            pred_len = self.predict_len1(mu)
            pred_len = self.predict_len2(pred_len)
        return mem, mu, logvar, pred_len

    def forward_w_attn(self, x, mask):
        "Forward pass that saves attention weights"
        attn_wts = []
        for i, attn_layer in enumerate(self.layers):
            x, wts = attn_layer(x, mask, return_attn=True)
            attn_wts.append(wts.detach().cpu())
        mem = self.norm(x)
        if self.bypass_bottleneck:
            mu, logvar = Variable(torch.tensor([0.0])), Variable(torch.tensor([0.0]))
        else:
            mem = mem.permute(0, 2, 1)
            mem = self.conv_bottleneck(mem)
            mem = mem.contiguous().view(mem.size(0), -1)
            mu, logvar = self.z_means(mem), self.z_var(mem)
            mem = self.reparameterize(mu, logvar, self.eps_scale)
            pred_len = self.predict_len1(mu)
            pred_len = self.predict_len2(pred_len)
        return mem, mu, logvar, pred_len, attn_wts

class EncoderLayer(nn.Module):
    "Self-attention/feedforward implementation"
    def __init__(self, size, src_len, self_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.src_len = src_len
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(self.size, dropout), 2)

    def forward(self, x, mask, return_attn=False):
        if return_attn:
            attn = self.self_attn(x, x, x, mask, return_attn=True)
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            return self.sublayer[1](x, self.feed_forward), attn
        else:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            return self.sublayer[1](x, self.feed_forward)

class VAEDecoder(nn.Module):
    "Base transformer decoder architecture"
    def __init__(self, encoder_layers, decoder_layers, N, d_latent, bypass_bottleneck):
        super().__init__()
        self.final_encodes = clones(encoder_layers, 1)
        self.layers = clones(decoder_layers, N)
        self.norm = LayerNorm(decoder_layers.size)
        self.bypass_bottleneck = bypass_bottleneck
        self.size = decoder_layers.size
        self.tgt_len = decoder_layers.tgt_len

        # Reshaping memory with deconvolution
        self.linear = nn.Linear(d_latent, 576)
        self.deconv_bottleneck = DeconvBottleneck(decoder_layers.size)

    def forward(self, x, mem, src_mask, tgt_mask):
        ### Deconvolutional bottleneck (up-sampling)
        if not self.bypass_bottleneck:
            mem = F.relu(self.linear(mem))
            mem = mem.view(-1, 64, 9)
            mem = self.deconv_bottleneck(mem)
            mem = mem.permute(0, 2, 1)
        ### Final self-attention layer
        for final_encode in self.final_encodes:
            mem = final_encode(mem, src_mask)
        # Batch normalization
        mem = self.norm(mem)
        ### Source-attention layers
        for i, attn_layer in enumerate(self.layers):
            x = attn_layer(x, mem, mem, src_mask, tgt_mask)
        return self.norm(x)

    def forward_w_attn(self, x, mem, src_mask, tgt_mask):
        "Forward pass that saves attention weights"
        if not self.bypass_bottleneck:
            mem = F.relu(self.linear(mem))
            mem = mem.view(-1, 64, 9)
            mem = self.deconv_bottleneck(mem)
            mem = mem.permute(0, 2, 1)
        for final_encode in self.final_encodes:
            mem, deconv_wts  = final_encode(mem, src_mask, return_attn=True)
        mem = self.norm(mem)
        src_attn_wts = []
        for i, attn_layer in enumerate(self.layers):
            x, wts = attn_layer(x, mem, mem, src_mask, tgt_mask, return_attn=True)
            src_attn_wts.append(wts.detach().cpu())
        return self.norm(x), [deconv_wts.detach().cpu()], src_attn_wts

class DecoderLayer(nn.Module):
    "Self-attention/source-attention/feedforward implementation"
    def __init__(self, size, tgt_len, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.tgt_len = tgt_len
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(self.size, dropout), 3)

    def forward(self, x, memory_key, memory_val, src_mask, tgt_mask, return_attn=False):
        m_key = memory_key
        m_val = memory_val
        if return_attn:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            src_attn = self.src_attn(x, m_key, m_val, src_mask, return_attn=True)
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m_key, m_val, src_mask))
            return self.sublayer[2](x, self.feed_forward), src_attn
        else:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m_key, m_val, src_mask))
            return self.sublayer[2](x, self.feed_forward)

############## Attention and FeedForward ################

class MultiHeadedAttention(nn.Module):
    "Multihead attention implementation (based on Vaswani et al.)"
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads"
        super().__init__()
        assert d_model % h == 0
        #We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, return_attn=False):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                            for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        if return_attn:
            return self.attn
        else:
            return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Feedforward implementation"
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


############## BOTTLENECKS #################

class ConvBottleneck(nn.Module):
    """
    Set of convolutional layers to reduce memory matrix to single
    latent vector
    """
    def __init__(self, size):
        super().__init__()
        conv_layers = []
        in_d = size
        first = True
        for i in range(3):
            out_d = int((in_d - 64) // 2 + 64)
            if first:
                kernel_size = 9
                first = False
            else:
                kernel_size = 8
            if i == 2:
                out_d = 64
            conv_layers.append(nn.Sequential(nn.Conv1d(in_d, out_d, kernel_size), nn.MaxPool1d(2)))
            in_d = out_d
        self.conv_layers = ListModule(*conv_layers)

    def forward(self, x):
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        return x

class DeconvBottleneck(nn.Module):
    """
    Set of deconvolutional layers to reshape latent vector
    back into memory matrix
    """
    def __init__(self, size):
        super().__init__()
        deconv_layers = []
        in_d = 64
        for i in range(3):
            out_d = (size - in_d) // 4 + in_d
            stride = 4 - i
            kernel_size = 11
            if i == 2:
                out_d = size
                stride = 1
            deconv_layers.append(nn.Sequential(nn.ConvTranspose1d(in_d, out_d, kernel_size,
                                                                  stride=stride, padding=2)))
            in_d = out_d
        self.deconv_layers = ListModule(*deconv_layers)

    def forward(self, x):
        for deconv in self.deconv_layers:
            x = F.relu(deconv(x))
        return x

############## Property Predictor #################

class PropertyPredictor(nn.Module):
    "Optional property predictor module"
    def __init__(self, d_pp, depth_pp, d_latent):
        super().__init__()
        prediction_layers = []
        for i in range(depth_pp):
            if i == 0:
                linear_layer = nn.Linear(d_latent, d_pp)
            elif i == depth_pp - 1:
                linear_layer = nn.Linear(d_pp, 1)
            else:
                linear_layer = nn.Linear(d_pp, d_pp)
            prediction_layers.append(linear_layer)
        self.prediction_layers = ListModule(*prediction_layers)

    def forward(self, x):
        for prediction_layer in self.prediction_layers:
            x = F.relu(prediction_layer(x))
        return x

############## Embedding Layers ###################

class Embeddings(nn.Module):
    "Transforms input token id tensors to size d_model embeddings"
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Static sinusoidal positional encoding layer"
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

############## Utility Layers ####################

class TorchLayerNorm(nn.Module):
    "Construct a layernorm module (pytorch)"
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.bn = nn.BatchNorm1d(features)

    def forward(self, x):
        return self.bn(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (manual)"
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size"
        return x + self.dropout(sublayer(self.norm(x)))


import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

# from transvae.tvae_util import *
# from transvae.opt import NoamOpt, AdamOpt
# from transvae.trans_models import VAEShell, Generator, ConvBottleneck, DeconvBottleneck, PropertyPredictor, Embeddings, LayerNorm

# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# Attention architectures inspired by the ^^^ implementation
########## Model Classes ############

class RNNAttn(VAEShell):
    """
    RNN-based VAE class with attention.
    """
    def __init__(self, params={}, name=None, N=3, d_model=128,
                 d_latent=128, dropout=0.1, tf=True,
                 bypass_attention=False,
                 bypass_bottleneck=False,
                 property_predictor=False,
                 d_pp=256, depth_pp=2,
                 load_fn=None):
        super().__init__(params, name)
        """
        Instatiating a GruaVAE object builds the model architecture, data structs
        to store the model parameters and training information and initiates model
        weights. Most params have default options but vocabulary must be provided.

        Arguments:
            params (dict, required): Dictionary with model parameters. Keys must match
                                     those written in this module
            name (str): Name of model (all save and log files will be written with
                        this name)
            N (int): Number of repeat encoder and decoder layers
            d_model (int): Dimensionality of model (embeddings and attention)
            d_latent (int): Dimensionality of latent space
            dropout (float): Rate of dropout
            bypass_bottleneck (bool): If false, model functions as standard autoencoder
        """

        ### Set learning rate for Adam optimizer
        if 'ADAM_LR' not in self.params.keys():
            self.params['ADAM_LR'] = 3e-4

        ### Store architecture params
        self.model_type = 'rnn_attn'
        self.params['model_type'] = self.model_type
        self.params['N'] = N
        self.params['d_model'] = d_model
        self.params['d_latent'] = d_latent
        self.params['dropout'] = dropout
        self.params['teacher_force'] = tf
        self.params['bypass_attention'] = bypass_attention
        self.params['bypass_bottleneck'] = bypass_bottleneck
        self.params['property_predictor'] = property_predictor
        self.params['d_pp'] = d_pp
        self.params['depth_pp'] = depth_pp
        self.arch_params = ['N', 'd_model', 'd_latent', 'dropout', 'teacher_force',
                            'bypass_attention', 'bypass_bottleneck', 'property_predictor',
                            'd_pp', 'depth_pp']

        ### Build model architecture
        if load_fn is None:
            self.build_model()
        else:
            self.load(load_fn)

    def build_model(self):
        """
        Build model architecture. This function is called during initialization as well as when
        loading a saved model checkpoint
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = RNNAttnEncoder(self.params['d_model'], self.params['d_latent'], self.params['N'],
                                 self.params['dropout'], self.src_len, self.params['bypass_attention'],
                                 self.params['bypass_bottleneck'], self.device)
        decoder = RNNAttnDecoder(self.params['d_model'], self.params['d_latent'], self.params['N'],
                                 self.params['dropout'], self.params['teacher_force'], self.params['bypass_bottleneck'],
                                 self.device)
        generator = Generator(self.params['d_model'], self.vocab_size)
        src_embed = Embeddings(self.params['d_model'], self.vocab_size)
        tgt_embed = Embeddings(self.params['d_model'], self.vocab_size)
        if self.params['property_predictor']:
            property_predictor = PropertyPredictor(self.params['d_pp'], self.params['depth_pp'], self.params['d_latent'])
        else:
            property_predictor = None
        self.model = RNNEncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator,
                                       property_predictor, self.params)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.model.cuda()
            self.params['CHAR_WEIGHTS'] = self.params['CHAR_WEIGHTS'].cuda()

        ### Initiate optimizer
        self.optimizer = AdamOpt([p for p in self.model.parameters() if p.requires_grad],
                                  self.params['ADAM_LR'], optim.Adam)

class RNN(VAEShell):
    """
    RNN-based VAE without attention.
    """
    def __init__(self, params={}, name=None, N=3, d_model=128,
                 d_latent=128, dropout=0.1, tf=True,
                 bypass_bottleneck=False, property_predictor=False,
                 d_pp=256, depth_pp=2, load_fn=None):
        super().__init__(params, name)

        ### Set learning rate for Adam optimizer
        if 'ADAM_LR' not in self.params.keys():
            self.params['ADAM_LR'] = 3e-4

        ### Store architecture params
        self.model_type = 'rnn'
        self.params['model_type'] = self.model_type
        self.params['N'] = N
        self.params['d_model'] = d_model
        self.params['d_latent'] = d_latent
        self.params['dropout'] = dropout
        self.params['teacher_force'] = tf
        self.params['bypass_bottleneck'] = bypass_bottleneck
        self.params['property_predictor'] = property_predictor
        self.params['d_pp'] = d_pp
        self.params['depth_pp'] = depth_pp
        self.arch_params = ['N', 'd_model', 'd_latent', 'dropout', 'teacher_force', 'bypass_bottleneck',
                            'property_predictor', 'd_pp', 'depth_pp']

        ### Build model architecture
        if load_fn is None:
            self.build_model()
        else:
            self.load(load_fn)

    def build_model(self):
        """
        Build model architecture. This function is called during initialization as well as when
        loading a saved model checkpoint
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = RNNEncoder(self.params['d_model'], self.params['d_latent'], self.params['N'],
                             self.params['dropout'], self.params['bypass_bottleneck'], self.device)
        decoder = RNNDecoder(self.params['d_model'], self.params['d_latent'], self.params['N'],
                             self.params['dropout'], 125, self.params['teacher_force'], self.params['bypass_bottleneck'],
                             self.device)
        generator = Generator(self.params['d_model'], self.vocab_size)
        src_embed = Embeddings(self.params['d_model'], self.vocab_size)
        tgt_embed = Embeddings(self.params['d_model'], self.vocab_size)
        if self.params['property_predictor']:
            property_predictor = PropertyPredictor(self.params['d_pp'], self.params['depth_pp'], self.params['d_latent'])
        else:
            property_predictor = None
        self.model = RNNEncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator,
                                       property_predictor, self.params)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.model.cuda()
            self.params['CHAR_WEIGHTS'] = self.params['CHAR_WEIGHTS'].cuda()

        ### Initiate optimizer
        self.optimizer = AdamOpt([p for p in self.model.parameters() if p.requires_grad],
                                  self.params['ADAM_LR'], optim.Adam)


########## Recurrent Sub-blocks ############

class RNNEncoderDecoder(nn.Module):
    """
    Recurrent Encoder-Decoder Architecture
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator,
                 property_predictor, params):
        super().__init__()
        self.params = params
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.property_predictor = property_predictor

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        mem, mu, logvar = self.encode(src)
        x, h = self.decode(tgt, mem)
        x = self.generator(x)
        if self.property_predictor is not None:
            prop = self.predict_property(mu)
        else:
            prop = None
        return x, mu, logvar, prop

    def encode(self, src):
        return self.encoder(self.src_embed(src))

    def decode(self, tgt, mem):
        return self.decoder(self.src_embed(tgt), mem)

    def predict_property(self, mu):
        return self.property_predictor(mu)

class RNNAttnEncoder(nn.Module):
    """
    Recurrent encoder with attention architecture
    """
    def __init__(self, size, d_latent, N, dropout, src_length, bypass_attention, bypass_bottleneck, device):
        super().__init__()
        self.size = size
        self.n_layers = N
        self.max_length = src_length+1
        self.bypass_attention = bypass_attention
        self.bypass_bottleneck = bypass_bottleneck
        self.device = device

        self.gru = nn.GRU(self.size, self.size, num_layers=N, dropout=dropout)
        self.attn = nn.Linear(self.size * 2, self.max_length)
        self.conv_bottleneck = ConvBottleneck(size)
        self.z_means = nn.Linear(576, d_latent)
        self.z_var = nn.Linear(576, d_latent)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, return_attn=False):
        h = self.initH(x.shape[0])
        x = x.permute(1, 0, 2)
        x_out, h = self.gru(x, h)
        x = x.permute(1, 0, 2)
        x_out = x_out.permute(1, 0, 2)
        mem = self.norm(x_out)
        if not self.bypass_attention:
            attn_weights = F.softmax(self.attn(torch.cat((x, mem), 2)), dim=2)
            attn_applied = torch.bmm(attn_weights, mem)
            mem = F.relu(attn_applied)
        if self.bypass_bottleneck:
            mu, logvar = Variable(torch.tensor([100.])), Variable(torch.tensor([100.]))
        else:
            mem = mem.permute(0, 2, 1)
            mem = self.conv_bottleneck(mem)
            mem = mem.contiguous().view(mem.size(0), -1)
            mu, logvar = self.z_means(mem), self.z_var(mem)
            mem = self.reparameterize(mu, logvar)
        if return_attn:
            return mem, mu, logvar, attn_weights.detach().cpu()
        else:
            return mem, mu, logvar

    def initH(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.size, device=self.device)

class RNNAttnDecoder(nn.Module):
    """
    Recurrent decoder with attention architecture
    """
    def __init__(self, size, d_latent, N, dropout, tf, bypass_bottleneck, device):
        super().__init__()
        self.size = size
        self.n_layers = N
        self.teacher_force = tf
        if self.teacher_force:
            self.gru_size = self.size * 2
        else:
            self.gru_size = self.size
        self.bypass_bottleneck = bypass_bottleneck
        self.device = device

        self.linear = nn.Linear(d_latent, 576)
        self.deconv_bottleneck = DeconvBottleneck(size)
        self.dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(self.gru_size, self.size, num_layers=N, dropout=dropout)
        self.norm = LayerNorm(size)

    def forward(self, tgt, mem):
        embedded = self.dropout(tgt)
        h = self.initH(mem.shape[0])
        if not self.bypass_bottleneck:
            mem = F.relu(self.linear(mem))
            mem = mem.contiguous().view(-1, 64, 9)
            mem = self.deconv_bottleneck(mem)
            mem = mem.permute(0, 2, 1)
            mem = self.norm(mem)
        mem = mem[:,:-1,:]
        if self.teacher_force:
            mem = torch.cat((embedded, mem), dim=2)
        mem = mem.permute(1, 0, 2)
        mem = mem.contiguous()
        x, h = self.gru(mem, h)
        x = x.permute(1, 0, 2)
        x = self.norm(x)
        return x, h

    def initH(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.size, device=self.device).float()

class RNNEncoder(nn.Module):
    """
    Simple recurrent encoder architecture
    """
    def __init__(self, size, d_latent, N, dropout, bypass_bottleneck, device):
        super().__init__()
        self.size = size
        self.n_layers = N
        self.bypass_bottleneck = bypass_bottleneck
        self.device = device

        self.gru = nn.GRU(self.size, self.size, num_layers=N, dropout=dropout)
        self.z_means = nn.Linear(size, d_latent)
        self.z_var = nn.Linear(size, d_latent)
        self.norm = LayerNorm(size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        h = self.initH(x.shape[0])
        x = x.permute(1, 0, 2)
        x, h = self.gru(x, h)
        mem = self.norm(h[-1,:,:])
        if self.bypass_bottleneck:
            mu, logvar = Variable(torch.tensor([0.0])), Variable(torch.tensor([0.0]))
        else:
            mu, logvar = self.z_means(mem), self.z_var(mem)
            mem = self.reparameterize(mu, logvar)
        return mem, mu, logvar

    def initH(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.size, device=self.device).float()

class RNNDecoder(nn.Module):
    """
    Simple recurrent decoder architecture
    """
    def __init__(self, size, d_latent, N, dropout, tgt_length, tf, bypass_bottleneck, device):
        super().__init__()
        self.size = size
        self.n_layers = N
        self.max_length = tgt_length+1
        self.teacher_force = tf
        if self.teacher_force:
            self.gru_size = self.size * 2
        else:
            self.gru_size = self.size
        self.bypass_bottleneck = bypass_bottleneck
        self.device = device

        self.gru = nn.GRU(self.gru_size, self.size, num_layers=N, dropout=dropout)
        self.unbottleneck = nn.Linear(d_latent, size)
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(size)

    def forward(self, tgt, mem):
        h = self.initH(mem.shape[0])
        embedded = self.dropout(tgt)
        if not self.bypass_bottleneck:
            mem = F.relu(self.unbottleneck(mem))
            mem = mem.unsqueeze(1).repeat(1, self.max_length, 1)
            mem = self.norm(mem)
        if self.teacher_force:
            mem = torch.cat((embedded, mem), dim=2)
        mem = mem.permute(1, 0, 2)
        mem = mem.contiguous()
        x, h = self.gru(mem, h)
        x = x.permute(1, 0, 2)
        x = self.norm(x)
        return x, h

    def initH(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.size, device=self.device)



import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from transvae.tvae_util import KLAnnealer

# Plotting functions

def plot_test_train_curves(paths, target_path=None, loss_type='tot_loss', data_type='test', labels=None, colors=None):
    """
    Plots the training curves for a set of model log files

    Arguments:
        paths (list, req): List of paths to log files (generated during training)
        target_path (str): Optional path to plot target loss (if you are trying to replicate or improve upon a given loss curve)
        loss_type (str): The type of loss to plot - tot_loss, kld_loss, recon_loss, etc.
        labels (list): List of labels for plot legend
        colors (list): List of colors for each training curve
    """
    if colors is None:
        colors = ['#005073', '#B86953', '#932191', '#90041F', '#0F4935']
    if labels is None:
        labels = []
        for path in paths:
            path = path.split('/')[-1].split('log_GRUGRU_')[-1].split('.')[0]
            labels.append(path)
    plt.figure(figsize=(10,8))
    ax = plt.subplot(111)

    for i, path in enumerate(paths):
        df = pd.read_csv(path)
        try:
            data = df[df.data_type == data_type].groupby('epoch').mean()[loss_type]
        except KeyError:
            data = df[df.data_type == data_type].groupby('epoch').mean()['bce_loss']
        if loss_type == 'kld_loss':
            klannealer = KLAnnealer(1e-8, 0.05, 60, 0)
            klanneal = []
            for j in range(60):
                klanneal.append(klannealer(j))
            data /= klanneal
        plt.plot(data, c=colors[i], lw=2.5, label=labels[i], alpha=0.95)
    if target_path is not None:
        df = pd.read_csv(target_path)
        try:
            target = df[df.data_type == data_type].groupby('epoch').mean()[loss_type]
        except KeyError:
            target = df[df.data_type == data_type].groupby('epoch').mean()['bce_loss']
        plt.plot(target, c='black', ls=':', lw=2.5, label='Approximate Goal')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.yscale('log')
    plt.ylabel(loss_type, rotation='horizontal', labelpad=30)
    plt.xlabel('epoch')
    return plt

def plot_loss_by_type(path, colors=None):
    """
    Plot the training curve of one model for each loss type

    Arguments:
        path (str, req): Path to log file of trained model
        colors (list): Colors for each loss type
    """
    if colors is None:
        colors = ['#005073', '#B86953', '#932191', '#90041F', '#0F4935']

    df = pd.read_csv(path)

    plt.figure(figsize=(10,8))
    ax = plt.subplot(111)

    loss_types = ['tot_loss', 'bce_loss', 'kld_loss', 'pred_loss']
    for i, loss_type in enumerate(loss_types):
        train_data = df[df.data_type == 'train'].groupby('epoch').mean()[loss_type]
        test_data = df[df.data_type == 'test'].groupby('epoch').mean()[loss_type]
        plt.plot(train_data, c=colors[i], label='train_'+loss_type)
        plt.plot(test_data, c=colors[i], label='test_'+loss_type, ls=':')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.yscale('log')
    plt.ylabel('Loss', rotation='horizontal')
    plt.xlabel('epoch')
    plt.title(path.split('/')[-1].split('log_GRUGRU_')[-1].split('.')[0])
    return plt

def plot_reconstruction_accuracies(dir, colors=None):
    """
    Plots token, SMILE and positional reconstruction accuracies for all model types in directory

    Arguments:
        dir (str, req): Directory to json files containing stored accuracies for each trained model
        colors (list): List of colors for each trained model
    """
    if colors is None:
        colors = ['#005073', '#B86953', '#932191', '#90041F', '#0F4935']

    data, labels = get_json_data(dir)

    smile_accs = {}
    token_accs = {}
    pos_accs = {}
    for k, v in data.items():
        smile_accs[k] = v['accs']['test'][0]
        token_accs[k] = v['accs']['test'][1]
        pos_accs[k] = v['accs']['test'][2]

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(12,4), sharey=True,
                                     gridspec_kw={'width_ratios': [1, 1, 2]})
    a0.bar(np.arange(len(smile_accs)), smile_accs.values(), color=colors[:len(smile_accs)])
    a0.set_xticks(np.arange(len(smile_accs)))
    a0.set_xticklabels(labels=smile_accs.keys(), rotation=45)
    a0.set_ylim([0,1])
    a0.set_ylabel('Accuracy', rotation=0, labelpad=30)
    a0.set_title('Per SMILE')
    a1.bar(np.arange(len(token_accs)), token_accs.values(), color=colors[:len(token_accs)])
    a1.set_xticks(np.arange(len(token_accs)))
    a1.set_xticklabels(labels=token_accs.keys(), rotation=45)
    a1.set_ylim([0,1])
    a1.set_title('Per Token')
    for i, set in enumerate(pos_accs.values()):
        a2.plot(set, lw=2, color=colors[i])
    a2.set_xlabel('Token Position')
    a2.set_ylim([0,1])
    a2.set_title('Per Token Sequence Position')
    return fig

def plot_moses_metrics(dir, colors=None):
    """
    Plots tiled barplot depicting the performance of the model on each MOSES metric as a function
    of epoch.

    Arguments:
        dir (str, req): Directory to json files containing calculated MOSES metrics for each model type
        colors (list): List of colors for each trained model

    """
    if colors is None:
        colors = ['#005073', '#B86953', '#932191', '#90041F', '#0F4935']

    data, labels = get_json_data(dir)
    data['paper_vae'] = {'valid': 0.977,
                         'unique@1000': 1.0,
                         'unique@10000': 0.998,
                         'FCD/Test': 0.099,
                         'SNN/Test': 0.626,
                         'Frag/Test': 0.999,
                         'Scaf/Test': 0.939,
                         'FCD/TestSF': 0.567,
                         'SNN/TestSF': 0.578,
                         'Frag/TestSF': 0.998,
                         'Scaf/TestSF': 0.059,
                         'IntDiv': 0.856,
                         'IntDiv2': 0.850,
                         'Filters': 0.997,
                         'logP': 0.121,
                         'SA': 0.219,
                         'QED': 0.017,
                         'weight': 3.63,
                         'Novelty': 0.695,
                         'runtime': 0.0}
    labels.append('paper_vae')
    metrics = list(data['paper_vae'].keys())

    fig, axs = plt.subplots(5, 4, figsize=(20,14))
    for i, ax in enumerate(fig.axes):
        metric = metrics[i]
        metric_data = []
        for label in labels:
            metric_data.append(data[label][metric])
        ax.bar(np.arange(len(metric_data)), metric_data, color=colors[:len(metric_data)])
        ax.set_xticks(np.arange(len(metric_data)))
        ax.set_xticklabels(labels=labels)
        ax.set_title(metric)
    return fig


def get_json_data(dir, fns=None, labels=None):
    """
    Opens and stores json data from a given directory

    Arguments:
        dir (str, req): Directory containing the json files
        labels (list): Labels corresponding to each file
    Returns:
        data (dict): Dictionary containing all data within
                     json files
        labels (list): List of keys corresponding to dictionary entries
    """
    if fns is None:
        fns = []
        for fn in os.listdir(dir):
            if '.json' in fn:
                fns.append(os.path.join(dir, fn))
    if labels is None:
        labels = []
        fn = fn.split('/')[-1].split('2milmoses_')[1].split('.json')[0].split('_')[0]
        labels.append(fn)

    data = {}
    for fn, label in zip(fns, labels):
        with open(fn, 'r') as f:
            dump = json.load(f)
        data[label] = dump
    return data, labels



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

# from transvae.tvae_util import *

def vae_data_gen(mols, props, char_dict):
    """
    Encodes input smiles to tensors with token ids

    Arguments:
        mols (np.array, req): Array containing input molecular structures
        props (np.array, req): Array containing scalar chemical property values
        char_dict (dict, req): Dictionary mapping tokens to integer id
    Returns:
        encoded_data (torch.tensor): Tensor containing encodings for each
                                     SMILES string
    """
    smiles = mols[:,0]
    if props is None:
        props = np.zeros(smiles.shape)
    del mols
    smiles = [tokenizer(x) for x in smiles]
    encoded_data = torch.empty((len(smiles), 128))
    for j, smi in enumerate(smiles):
        encoded_smi = encode_smiles(smi, 126, char_dict)
        encoded_smi = [0] + encoded_smi
        encoded_data[j,:-1] = torch.tensor(encoded_smi)
        encoded_data[j,-1] = torch.tensor(props[j])
    return encoded_data

def make_std_mask(tgt, pad):
    """
    Creates sequential mask matrix for target input (adapted from
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)

    Arguments:
        tgt (torch.tensor, req): Target vector of token ids
        pad (int, req): Padding token id
    Returns:
        tgt_mask (torch.tensor): Sequential target mask
    """
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask


import argparse
# from transvae.trans_models import TransVAE
# from transvae.rnn_models import RNN, RNNAttn

def model_init(args, params={}):
    ### Model Name
    if args.save_name is None:
        if args.model == 'transvae':
            save_name = 'trans{}x-{}_{}'.format(args.d_feedforward // args.d_model,
                                                args.d_model,
                                                args.data_source)
        else:
            save_name = '{}-{}_{}'.format(args.model,
                                          args.d_model,
                                          args.data_source)
    else:
        save_name = args.save_name

    ### Load Model
    if args.model == 'transvae':
        vae = TransVAE(params=params, name=save_name, d_model=args.d_model,
                       d_ff=args.d_feedforward, d_latent=args.d_latent,
                       property_predictor=args.property_predictor, d_pp=args.d_property_predictor,
                       depth_pp=args.depth_property_predictor)
    elif args.model == 'rnnattn':
        vae = RNNAttn(params=params, name=save_name, d_model=args.d_model,
                      d_latent=args.d_latent, property_predictor=args.property_predictor,
                      d_pp=args.d_property_predictor, depth_pp=args.depth_property_predictor)
    elif args.model == 'rnn':
        vae = RNN(params=params, name=save_name, d_model=args.d_model,
                  d_latent=args.d_latent, property_predictor=args.property_predictor,
                  d_pp=args.d_property_predictor, depth_pp=args.depth_property_predictor)

    return vae

def train_parser():
    parser = argparse.ArgumentParser()
    ### Architecture Parameters
    parser.add_argument('--model', choices=['transvae', 'rnnattn', 'rnn'],
                        required=True, type=str)
    parser.add_argument('--d_model', default=128, type=int)
    parser.add_argument('--d_feedforward', default=128, type=int)
    parser.add_argument('--d_latent', default=128, type=int)
    parser.add_argument('--property_predictor', default=False, action='store_true')
    parser.add_argument('--d_property_predictor', default=256, type=int)
    parser.add_argument('--depth_property_predictor', default=2, type=int)
    ### Hyperparameters
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--batch_chunks', default=5, type=int)
    parser.add_argument('--beta', default=0.05, type=float)
    parser.add_argument('--beta_init', default=1e-8, type=float)
    parser.add_argument('--anneal_start', default=0, type=int)
    parser.add_argument('--adam_lr', default=3e-4, type=float)
    parser.add_argument('--lr_scale', default=1, type=float)
    parser.add_argument('--warmup_steps', default=10000, type=int)
    parser.add_argument('--eps_scale', default=1, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    ### Data Parameters
    parser.add_argument('--data_source', choices=['zinc', 'pubchem', 'custom'],
                        required=True, type=str)
    parser.add_argument('--train_mols_path', default=None, type=str)
    parser.add_argument('--test_mols_path', default=None, type=str)
    parser.add_argument('--train_props_path', default=None, type=str)
    parser.add_argument('--test_props_path', default=None, type=str)
    parser.add_argument('--vocab_path', default=None, type=str)
    parser.add_argument('--char_weights_path', default=None, type=str)
    ### Load Parameters
    parser.add_argument('--checkpoint', default=None, type=str)
    ### Save Parameters
    parser.add_argument('--save_name', default=None, type=str)
    parser.add_argument('--save_freq', default=5, type=int)

    return parser

def sample_parser():
    parser = argparse.ArgumentParser()
    ### Load Files
    parser.add_argument('--model', choices=['transvae', 'rnnattn', 'rnn'],
                        required=True, type=str)
    parser.add_argument('--model_ckpt', required=True, type=str)
    parser.add_argument('--mols', default=None, type=str)
    ### Sampling Parameters
    parser.add_argument('--sample_mode', choices=['rand', 'high_entropy', 'k_high_entropy'],
                        required=True, type=str)
    parser.add_argument('--k', default=15, type=int)
    parser.add_argument('--condition', default='', type=str)
    parser.add_argument('--entropy_cutoff', default=5, type=float)
    parser.add_argument('--n_samples', default=30000, type=int)
    parser.add_argument('--n_samples_per_batch', default=100, type=int)
    ### Save Parameters
    parser.add_argument('--save_path', default=None, type=str)

    return parser

def attn_parser():
    parser = argparse.ArgumentParser()
    ### Load Files
    parser.add_argument('--model', choices=['transvae', 'rnnattn'],
                        required=True, type=str)
    parser.add_argument('--model_ckpt', required=True, type=str)
    parser.add_argument('--mols', required=True, type=str)
    ### Sampling Parameters
    parser.add_argument('--n_samples', default=5000, type=int)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--batch_chunks', default=5, type=int)
    parser.add_argument('--shuffle', default=False, action='store_true')
    ### Save Parameters
    parser.add_argument('--save_path', default=None, type=str)

    return parser

def vocab_parser():
    parser = argparse.ArgumentParser()
    ### Vocab Parameters
    parser.add_argument('--mols', required=True, type=str)
    parser.add_argument('--freq_penalty', default=0.5, type=float)
    parser.add_argument('--pad_penalty', default=0.1, type=float)
    parser.add_argument('--vocab_name', default='custom_char_dict', type=str)
    parser.add_argument('--weights_name', default='custom_char_weights', type=str)
    parser.add_argument('--save_dir', default='data', type=str)

    return parser

