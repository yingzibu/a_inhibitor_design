"""
Date: 08-17-2023

SELIFES representation. 
Reference: https://github.com/aspuru-guzik-group/selfies
"""

from tqdm import tqdm
import selfies as sf
import pandas as pd

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

def SelfiesToDataset(selfies_df, max_len, savename=None):
    """
    Convert selfies into dataset. Dictionary
    dict_ = {'labels': labels, 
         'one_hots': one_hots,
         'alphabet': alphabet}
    :param selfies_df: dataframe of selfies
    :param savename: save file name 
    Return a dictionary containing 
        labels (len_dataset, max_len)
        onehots (len_dataset, max_len, len_alphabet)
        alphabet: a list of tokens
    """
    max_len_in_dataset = max(sf.len_selfies(s) for s in dataset)
    print('max len in dataset:', max_len_in_dataset)
    if max_len < max_len_in_dataset:    
        print(f'current defined max len: ',
              f'{max_len} < {max_len_in_dataset}, Abort')
        return
    else:
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
        if savename != None and name.split('.')[-1] == 'pt': 
            torch.save(dict_, savename)
            print('dataset saved at:', savename)
        return dict_
        
      