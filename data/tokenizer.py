"""
Date: 08-24-2023
Smiles tokenizer adpated from gentrl github 
https://github.com/insilicomedicine/GENTRL/blob/master/gentrl/tokenizer.py

cd /content/drive/MyDrive/JAK_ML/gentrl/
"""
import torch
import re
from tqdm import tqdm

_atoms = ['He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'Ar',
          'Ca', 'Ti', 'Cr', 'Fe', 'Ni', 'Cu', 'Ga', 'Ge', 'As', 'Se',
          'Br', 'Kr', 'Rb', 'Sr', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
          'Pd', 'Ag', 'Cd', 'Sb', 'Te', 'Xe', 'Ba', 'La', 'Ce', 'Pr',
          'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Er', 'Tm', 'Yb',
          'Lu', 'Hf', 'Ta', 'Re', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb',
          'Bi', 'At', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'Pu', 'Am', 'Cm',
          'Bk', 'Cf', 'Es', 'Fm', 'Md', 'Lr', 'Rf', 'Db', 'Sg', 'Mt',
          'Ds', 'Rg', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

ATOM_MAX_LEN = 120

def get_tokenizer_re(atoms):
    return re.compile('('+'|'.join(atoms)+r'|\%\d\d|.)')

_atoms_re = get_tokenizer_re(_atoms)

__i2t = {
    0: 'unused', 1: '>', 2: '<', 3: '2', 4: 'F', 5: 'Cl', 6: 'N',
    7: '[', 8: '6', 9: 'O', 10: 'c', 11: ']', 12: '#',
    13: '=', 14: '3', 15: ')', 16: '4', 17: '-', 18: 'n',
    19: 'o', 20: '5', 21: 'H', 22: '(', 23: 'C',
    24: '1', 25: 'S', 26: 's', 27: 'Br' , 28: '@', 29: '/', 30: '.', 
    31: 'P', 32: '+', 33: 'I', 34: 'Si', 35: 'B', 36: '\\', 
    37: '7', 38: '8', 39: '9'
}
# 28: '@', 29: '/', 30: '.'

__t2i = {
    '>': 1, '<': 2, '2': 3, 'F': 4, 'Cl': 5, 'N': 6, '[': 7, '6': 8,
    'O': 9, 'c': 10, ']': 11, '#': 12, '=': 13, '3': 14, ')': 15,
    '4': 16, '-': 17, 'n': 18, 'o': 19, '5': 20, 'H': 21, '(': 22,
    'C': 23, '1': 24, 'S': 25, 's': 26, 'Br': 27 , '@': 28, '/': 29, '.': 30, 
    'P': 31, '+': 32, 'I': 33, 'Si': 34, 'B': 35, '\\': 36,
    '7': 37, '8': 38, '9': 39 
}
#, '@': 28, '/': 29, '.': 30

def smiles_tokenizer(line, atoms=['Cl', 'Br', 'Si']):
    """
    Tokenizes SMILES string atom-wise using regular expressions. While this
    method is fast, it may lead to some mistakes: Sn may be considered as Tin
    or as Sulfur with Nitrogen in aromatic cycle. Because of this, you should
    specify a set of two-letter atoms explicitly.

    Parameters:
         atoms: set of two-letter atoms for tokenization
    """
    if atoms is not None:
        reg = get_tokenizer_re(atoms)
    else:
        reg = _atoms_re
    return reg.split(line)[1::2]


def encode(sm_list, pad_size=ATOM_MAX_LEN):
    """
    Encoder list of smiles to tensor of tokens
    """
    res = []
    lens = []
    # print(sm_list)
    for s in sm_list:
        tokens = ([1] + [__t2i[tok]
                for tok in smiles_tokenizer(s)])[:pad_size - 1]
        lens.append(len(tokens))
        tokens += (pad_size - len(tokens)) * [2]
        res.append(tokens)
    
    return torch.tensor(res).long(), lens

def encode_single(smi, pad_size=ATOM_MAX_LEN):
    tokens = ([1] + [__t2i[tok]
                for tok in smiles_tokenizer(smi)])[:pad_size - 1]
    len = len(tokens)
    tokens += (pad_size-len(tokens))*[2]
    return tokens, len
          
def decode(tokens_tensor):
    """
    Decodes from tensor of tokens to list of smiles
    """

    smiles_res = []

    for i in range(tokens_tensor.shape[0]):
        cur_sm = ''
        for t in tokens_tensor[i].detach().cpu().numpy():
            if t == 2:
                break
            elif t > 2:
                cur_sm += __i2t[t]

        smiles_res.append(cur_sm)

    return smiles_res


def get_vocab_size():
    return len(__i2t)

def To_matrix(t: torch.Tensor, vocab_size=get_vocab_size()):
    """
    max len = 120 as default
    Convert torch.Tensor of size (len_mol, max_len) to 
    torch.Tensor of (len_mol, vocab_size, max_len)
    :param t: tensor of size (len_mol, max_len) 
    :return (len_mol, max_len, vocab_size)
    """
    # print(t.shape)
    assert len(t.size()) == 2 # still vector

    output = torch.zeros([t.shape[0], t.shape[1], vocab_size])
    
    for i in range(t.shape[0]):
        temp = t[i] # (max_len), one drug
        for j, value in enumerate(temp): 
            output[i][j][value] = 1

    device_t = t.device.type

    output = output.to(device_t)
    
    return output

def To_vector(mat: torch.Tensor):
    """
    :param mat: (len_mol, max_len, vocab_size)
    : Return back to array
    """
    len_mol, max_len, vocab_size = mat.shape
    output = torch.zeros([len_mol, max_len])

    for i in range(mat.shape[0]):
        temp = mat[i] 
        # print(temp.shape) [max_len, vocab_size]
        pos = torch.argmax(temp, dim=1) # size [120]
        # print(pos.shape) 

        output[i] = pos
    
    device_mat = mat.device.type
    output = output.to(device_mat)
    return output
          
import rdkit
from rdkit import Chem
from IPython.display import display
m = Chem.MolFromSmiles
def remove_ions(smis:list,
    ion_list = ['[K+]', '[Li+]', '[Na+]', '[I-]', '[Cl-]', '[Br-]', 'Cl'], 
                print_info = False):
    ions_exist = False
    new_ion_list = []
    for ion in ion_list:
        ion1 = '.'+ ion
        ion2 = ion + '.'
        new_ion_list.append(ion1)
        new_ion_list.append(ion2)
    
    new_smis = []

    for smi in tqdm(smis): 
        ions_exist = False
        new_smi = ''
        for ion in new_ion_list:

            if ion in smi:
                ions_exist = True
                
                # ions_exist = True
                s = smi.replace(ion, '')
                if print_info == True: print('delete ion: ', ion)
                try:
                    mol = m(s)
                    new_smi = s
                except: new_smi = smi
        if ions_exist == False: new_smi = smi
        new_smis.append(new_smi)
    return new_smis
def delete_ions_strict(input_smiles:list): 
    print('delete ions 5 round: ')
    new_list = remove_ions(input_smiles)
    new_list = remove_ions(new_list)
    new_list = remove_ions(new_list)
    new_list = remove_ions(new_list)
    new_list = remove_ions(new_list)

    smile_ions = []
    smile_ions_inds = []
    new_list_new = []

    import random
    for j, i in enumerate(new_list): 
        if '.' in i:
            temp_list = i.split('.')
            temp_lens = [len(h) for h in temp_list]
            smile_ions.append(i)
            smile_ions_inds.append(j)
            new_list_new.append(temp_list[np.argmax(temp_lens)])
        else: new_list_new.append(i)
    print('still ions exist: ', len(smile_ions), 'will random display 20')
    show_num = min(len(smile_ions), 20)
    selected_smiles = random.choices(smile_ions, k=show_num)

    for j, i in enumerate(selected_smiles):
        print(f'compound {j}')
        temp_ind = smile_ions_inds[j]
        print(f'previous ind {temp_ind}')
        display(m(input_smiles[temp_ind]))
        print('simple deletion')
        display(m(new_list[temp_ind]))
        print('select the longest')
        display(m(new_list_new[temp_ind]))

    return new_list_new 
# new_list = remove_ions(smiles)
# new_list = remove_ions(new_list)
# new_list = remove_ions(new_list)
# new_list = remove_ions(new_list)
# new_list = remove_ions(new_list)

# ions_ex = []
# new_list_new = []
# super_strange = []
# super_strange_ind = []
# for j, i in enumerate(new_list):
#     if '.' in i:
#         temp_list = i.split('.')
#         temp_lens = [len(h) for h in temp_list]
        
#         print(j) 
#         super_strange_ind.append(j)
#         super_strange.append(i)

#         new_list_new.append(temp_list[np.argmax(temp_lens)])
#     else: new_list_new.append(i)
# print(len(ions_ex))

# from IPython.display import display
# for j, i in enumerate(super_strange[:20]): 
#     print(j)
#     display(m(big_df.iloc[super_strange_ind[j]]['smiles']))
#     # display(m(i))
#     display(m(new_list_new[super_strange_ind[j]]))
