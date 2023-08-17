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
            valid_sfs.append(smi)
        except sf.EncoderError: pass
        except sf.DecoderError: pass
    selfies_df = pd.DataFrame(valid_sfs, columns=['Smiles'])
    return selfies_df
      
  
