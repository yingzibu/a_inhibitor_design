"""
Date: 08-17-2023

SELIFES representation. 
Reference: https://github.com/aspuru-guzik-group/selfies
"""

from tqdm import tqdm
import selfies as sf
import pandas as pd
import torch
import os
import pickle

def SmilesToSelfies(smiles_df): 
    """
    Convert a list or DataFrame of smiles into selfies
    :param smiles_df: # input: DataFrame or list
    : return selfies of dataframe
    """
    valid_sfs = []
    for smi in tqdm(smiles_df, total=len(smiles_df)):
        try:
            drug_sf = sf.encoder(smi)
            drug_smi = sf.decoder(drug_sf)
            valid_sfs.append(drug_sf)
        except sf.EncoderError: pass
        except sf.DecoderError: pass
    selfies_df = pd.DataFrame(valid_sfs, columns=['Selfies'])
    return selfies_df


def SelfiesToDataset(selfies_df, max_len, savename=None, delete_long=True):
    """
    Convert selfies into dataset. Dictionary
    dict_ = {'labels': labels, 
         'one_hots': one_hots,
         'alphabet': alphabet}
    :param selfies_df: dataframe of selfies
    :param savename: save file name 
    :param delete_long: 
        True: delete long selfies if exceend max_len
        False: will truncate selfies to max_len

    Return a dictionary containing 
        labels (len_dataset, max_len)
        onehots (len_dataset, max_len, len_alphabet)
        alphabet: a list of tokens
    """
    dataset = []
    if isinstance(selfies_df, pd.DataFrame): 
        dataset = selfies_df['Selfies'].tolist()
    else: dataset = selfies_df # dataset type: list
        
    max_len_in_dataset = max(sf.len_selfies(s) for s in dataset)
    print('max len in dataset:', max_len_in_dataset)
    if max_len < max_len_in_dataset:    
        print(f'current defined max len: ',
              f'{max_len} < {max_len_in_dataset}')
        if delete_long:
            print('delete long selfies')
            new_dataset = [s for s in dataset if sf.len_selfies(s)<= max_len]
            print('delete #', len(dataset) - len(new_dataset))
            dataset = new_dataset
            
    alphabet = sf.get_alphabet_from_selfies(dataset)
    alphabet.add("[nop]")
    alphabet.add('.')
    alphabet = list(sorted(alphabet))
    print('len alphabet: ', len(alphabet))
    __t2i_sf = {s: i for i, s in enumerate(alphabet)}
    
    labels = []
    one_hots = []
    for selfi in tqdm(dataset, total=len(dataset)): 
        label, one_hot = sf.selfies_to_encoding(
            selfies = selfi, vocab_stoi = __t2i_sf, 
            pad_to_len = max_len, enc_type='both')
        labels.append(label)
        one_hots.append(one_hot)

    labels = torch.Tensor(labels).long() 
    one_hots = torch.Tensor(one_hots).long()
    dict_ = {'labels': labels, 
            'one_hots': one_hots,
            'alphabet': alphabet}
    if savename != None and savename.split('.')[-1] == 'pkl': 
        # torch.save(dict_, savename)
        with open(savename, 'wb) as f:
            pickle.dump(dict_, f)
        print('dataset saved at:', os.getcwd() + '/' + savename)
    if save
    return dict_


