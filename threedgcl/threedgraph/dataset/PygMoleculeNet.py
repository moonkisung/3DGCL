import os
import os.path as osp
import re
from tqdm import tqdm

from sklearn.utils import shuffle
import numpy as np

import torch
from torch_geometric.data import (Dataset, DataLoader, InMemoryDataset, Data, download_url, extract_gz)
import rdkit.Chem.AllChem as AllChem


x_map = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}


class MoleculeNet(InMemoryDataset):
    r"""The `MoleculeNet <http://moleculenet.ai/datasets-1>`_ benchmark
    collection  from the `"MoleculeNet: A Benchmark for Molecular Machine
    Learning" <https://arxiv.org/abs/1703.00564>`_ paper, containing datasets
    from physical chemistry, biophysics and physiology.
    All datasets come with the additional node and edge features introduced by
    the `Open Graph Benchmark <https://ogb.stanford.edu/docs/graphprop/>`_.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"ESOL"`,
            :obj:`"FreeSolv"`, :obj:`"Lipo"`, :obj:`"PCBA"`, :obj:`"MUV"`,
            :obj:`"HIV"`, :obj:`"BACE"`, :obj:`"BBPB"`, :obj:`"Tox21"`,
            :obj:`"ToxCast"`, :obj:`"SIDER"`, :obj:`"ClinTox"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{}'

    # Format: name: [display_name, url_name, csv_name, smiles_idx, y_idx]
    names = {
        'esol': ['ESOL', 'delaney-processed.csv', 'delaney-processed', -1, -2],
        'freesolv': ['FreeSolv', 'SAMPL.csv', 'SAMPL', 1, 2],
        'lipo': ['Lipophilicity', 'Lipophilicity.csv', 'Lipophilicity', 2, 1],
        'pcba': ['PCBA', 'pcba.csv.gz', 'pcba', -1,
                 slice(0, 128)],
        'muv': ['MUV', 'muv.csv.gz', 'muv', -1,
                slice(0, 17)],
        'hiv': ['HIV', 'HIV.csv', 'HIV', 0, -1],
        'bace': ['BACE', 'bace.csv', 'bace', 0, 2],
        'bbbp': ['BBPB', 'BBBP.csv', 'BBBP', -1, -2],
        'tox21': ['Tox21', 'tox21.csv.gz', 'tox21', -1,
                  slice(0, 12)],
        'toxcast': ['ToxCast', 'toxcast_data.csv.gz', 'toxcast_data', 0, slice(1, 618)],
        'sider': ['SIDER', 'sider.csv.gz', 'sider', 0,
                  slice(1, 28)],
        'clintox': ['ClinTox', 'clintox.csv.gz', 'clintox', 0, 1],
        'qm9': ['QM9', 'qm9.csv', 'qm9', 1,
                  slice(1, 28)],
        'moleculenet_except_sider': ['Moleculenet_Except_Sider', 'moleculenet_except_sider.csv', 'moleculenet_except_sider', 0, 1],
        'moleculenet': ['Moleculenet', 'moleculenet.csv', 'moleculenet', 0, 1],
        'pubchem01': ['Pubchem01', 'pubchem01.csv', 'pubchem01', 0, 1],
        'pubchem0102': ['Pubchem0102', 'pubchem0102.csv', 'pubchem0102', 0, 1],
        'pubchem010203': ['Pubchem010203', 'pubchem010203.csv', 'pubchem010203', 0, 1]
    }

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = name.lower()
        assert self.name in self.names.keys()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.names[self.name][2]}.csv'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        url = self.url.format(self.names[self.name][1])
        path = download_url(url, self.raw_dir)
        if self.names[self.name][1][-2:] == 'gz':
            extract_gz(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        from rdkit import Chem

        with open(self.raw_paths[0], 'r') as f:
            dataset = f.read().split('\n')[1:-1]
            dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

        data_list = []
        for line in tqdm(dataset):
            line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
            line = line.split(',')

            smiles = line[self.names[self.name][3]]
            
            ys = line[self.names[self.name][4]]
            #ys = ys.replace(0, -1)
            #ys = ys.fillna(0)
            ys = ys if isinstance(ys, list) else [ys]
            
            ys = [float(y) if len(y) > 0 else float('NaN') for y in ys]
            y = torch.tensor(ys, dtype=torch.float).view(-1)
            #print('1', y)
            #### classification ####
            #y[y==0] = -1
            #y[y!=y] = 0
            #print('2', y)
            if smiles.count('%') > 12:
                print(line, smiles)
                continue
                
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            mol = Chem.AddHs(mol)
            n_atoms = len(mol.GetAtoms()) # ????????? ?????? ?????? 1?????? skip
            if n_atoms==1:
                continue

            # Get Atom Number    
            zs = []
            for atom in mol.GetAtoms():
                z = []
                z.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
                zs.append(z)
            z = torch.tensor(zs, dtype=torch.long).view(-1)
            
            # Get Atomic Position
            ## ETKDG ##
            #AllChem.EmbedMultipleConfs(mol, numConfs=1000, pruneRmsThresh=0.1, randomSeed=2222, numThreads=0)
            #print('before ETKDG')
            
            try:
                AllChem.EmbedMultipleConfs(mol, numConfs=5, randomSeed=2222, numThreads=0)
            except: continue
            li = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)  # MMFF??? ??????
            li = [t[1] for t in li]
            sortidx = torch.argsort(torch.tensor(li))
            minidx = int(sortidx[0])
            maxidx1, maxidx2, maxidx3, maxidx4 = int(sortidx[-1]), int(sortidx[-2]), int(sortidx[-3]), int(sortidx[-4])
            minpos = mol.GetConformer(minidx).GetPositions()
            max1pos = mol.GetConformer(maxidx1).GetPositions()
            max2pos = mol.GetConformer(maxidx2).GetPositions()
            max3pos = mol.GetConformer(maxidx3).GetPositions()
            max4pos = mol.GetConformer(maxidx4).GetPositions()
            
            minpos_mmff = torch.tensor(minpos, dtype=torch.float)
            max1pos_mmff = torch.tensor(max1pos, dtype=torch.float)
            max2pos_mmff = torch.tensor(max2pos, dtype=torch.float)
            max3pos_mmff = torch.tensor(max3pos, dtype=torch.float)
            max4pos_mmff = torch.tensor(max4pos, dtype=torch.float)
            # atom ????????? ????????? ???????????? ?????? ???
            if 0.0 in minpos_mmff:
                continue
                
            data = Data(pos=minpos_mmff, max1pos_mmff=max1pos_mmff, max2pos_mmff=max2pos_mmff, max3pos_mmff=max3pos_mmff, max4pos_mmff=max4pos_mmff,
                        z=z, y=y,smiles=smiles)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

       
        
    def get_idx_split(self, task, data_size, train_size, valid_size, seed):
            if task =='esol':
                train_size, valid_size = 900, 100
            if task =='freesolv':
                train_size, valid_size = 500, 70
                
            ids = shuffle(range(data_size), random_state=seed)
            train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
            split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
            return split_dict
        
    def __repr__(self) -> str:
        return f'{self.names[self.name][0]}({len(self)})'
