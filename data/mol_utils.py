"""
Date: 08-17-2023

Reference: 
https://github.com/Yimeng-Wang/JAK-MTATFP/blob/main/preprocess.py
"""

# ! pip install rdkit --quiet
# ! pip install molvs --quiet

from rdkit import Chem
from molvs.normalize import Normalizer, Normalization
from rdkit.Chem.SaltRemover import SaltRemover
from molvs.charge import Reionizer, Uncharger
import torch

def preprocess(smi):
    mol = Chem.MolFromSmiles(smi)
    normalizer = Normalizer()
    new1 = normalizer.normalize(mol)
    remover = SaltRemover()
    new2 = remover(new1)
    neutralize1 = Reionizer()
    new3 = neutralize1(new2)
    neutralize2 = Uncharger()
    new4 = neutralize2(new3)
    new_smiles = Chem.MolToSmiles(new4, kekuleSmiles=False)
    return new_smiles
    
def onehot(k):
    """
    Converts a number to its torch.Size([k])
    one-hot representation vector
    :param k: (int) length of vector
    : return onehot function
    """
    def encode(label):
        y = torch.zeros(k)
        if label < k: y[label] = 1
        return y
    return encode # torch.Size([k])


# smi = 'C/C=C/C(O)=Nc1cccc(CNc2c(C(=N)O)cnn3cccc23)c1'
# new_smiles = preprocess(smi)
# print(new_smiles)
