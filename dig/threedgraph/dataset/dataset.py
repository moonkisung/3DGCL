import csv
import math
import logging
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx

from tqdm import tqdm
from tqdm import trange

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

import torch
from torch.utils import data as torch_data
from torch_geometric.data import InMemoryDataset, Data

from dig.sslgraph.utils.seed import setup_seed


def to_scaffold(smiles, chirality=True):
    """
    Return a scaffold SMILES string of this molecule.
    Parameters:
        chirality (bool, optional): consider chirality in the scaffold or not
    Returns:
        str
    """
    #smiles = self.to_smiles()
#     print(f'chirality={chirality}')
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles, includeChirality=chirality)
    return scaffold


def key_split(dataset, keys, lengths=None, key_lengths=None):

    def round_to_boundary(i):
        for j in range(min(i, len(dataset) - i)):
            if keys[indexes[i - j]] != keys[indexes[i - j - 1]]:
                return i - j
            if keys[indexes[i + j]] != keys[indexes[i + j - 1]]:
                return i + j
        if i < len(dataset) - i:
            return 0
        else:
            return len(dataset)

    keys = torch.as_tensor(keys)
    key_set, keys = torch.unique(keys, return_inverse=True)
    perm = torch.randperm(len(key_set))
    keys = perm[keys]
    indexes = keys.argsort().tolist()

    if key_lengths is not None:
        assert lengths is None
        key2count = keys.bincount()
        key_offset = 0
        lengths = []
        for key_length in key_lengths:
            lengths.append(key2count[key_offset: key_offset + key_length].sum().item())
            key_offset += key_length

    offset = 0
    offsets = [offset]
    for length in lengths:
        offset = round_to_boundary(offset + length)
        offsets.append(offset)
    offsets[-1] = len(dataset)
    #print('offsets', offsets)
    return [torch_data.Subset(dataset, indexes[offsets[i]: offsets[i + 1]]) for i in range(len(lengths))]
    #return [Data(dataset, indexes[offsets[i]: offsets[i + 1]]) for i in range(len(lengths))]


def scaffold_split(dataset, seed):
    """
    Randomly split a dataset into new datasets with non-overlapping scaffolds.
    Parameters:
        dataset (Dataset): dataset to split
        lengths (list of int): expected length for each split.
            Note the results may be different in length due to rounding.
    """
    setup_seed(seed)
    scaffold2id = {}
    keys = []
    #print(dataset.data)
    for sample in dataset:
        #print(sample)
        scaffold = to_scaffold(sample["smiles"])
        if scaffold not in scaffold2id:
            id = len(scaffold2id)
            scaffold2id[scaffold] = id
        else:
            id = scaffold2id[scaffold]
        keys.append(id)
    
    frac_train, frac_valid, frac_test = 0.8, 0.1, 0.1
    train_cutoff = int(frac_train * len(dataset))
    valid_cutoff = int(frac_valid * len(dataset))
    test_cutoff = int(frac_test * len(dataset))
    lengths = [train_cutoff, valid_cutoff, test_cutoff]
    #print(train_cutoff)
    #print(valid_cutoff)
    #print(test_cutoff)
    return key_split(dataset, keys, lengths)


def ordered_scaffold_split(dataset, lengths, chirality=True):
    """
    Split a dataset into new datasets with non-overlapping scaffolds and sorted w.r.t. number of each scaffold.

    Parameters:
        dataset (Dataset): dataset to split
        lengths (list of int): expected length for each split.
            Note the results may be different in length due to rounding.
    """
    frac_train, frac_valid, frac_test = 0.8, 0.1, 0.1
    
    scaffold2id = defaultdict(list)
    for idx, smiles in enumerate(dataset.data.smiles):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=chirality)
        #print(scaffold)
        scaffold2id[scaffold].append(idx)

    scaffold2id = {key: sorted(value) for key, value in scaffold2id.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffold2id.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    train_cutoff = frac_train * len(dataset)
    valid_cutoff = (frac_train + frac_valid) * len(dataset)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)
    #print('train_idx', train_idx)
    #print('valid_idx', valid_idx)
    #print('test_idx', test_idx)
    return torch_data.Subset(dataset, train_idx), torch_data.Subset(dataset, valid_idx), torch_data.Subset(dataset, test_idx)


def smiles2num(train):
    num_list = []
    pro_list = []
    for i, smiles in tqdm(enumerate(train['smiles'])):
        try:
            m = Chem.MolFromSmiles(smiles)
        except: continue
        try:
            num_atom = m.GetNumAtoms()
        except: 
            print(i, smiles)
            pro_list.append(i)
            continue
        num_list.append(num_atom)
    return num_list, pro_list

def smi_to_csv(filenum):  
    #'dataset/pubchem/pubchem/shard_01.smi'
    df = pd.read_csv(f'dataset/pubchem/pubchem/shard_{filenum}.smi', header=None)  # 1 파일 당 백만
    df.columns = ['smiles']
    num_list, pro_list = smiles2num(df)
    df = df.drop(pro_list)
    df['num'] = num_list
    df.to_csv(f'dataset/pubchem/pubchem/shard_{filenum}.csv', index=False)
    df = pd.read_csv(f'dataset/pubchem/pubchem/shard_{filenum}.csv')
    return df