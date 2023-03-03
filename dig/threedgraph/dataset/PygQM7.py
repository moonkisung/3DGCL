import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from sklearn.utils import shuffle

from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Data, DataLoader

import os
import sys
from typing import Callable, List, Optional

import torch.nn.functional as F
from torch_scatter import scatter
from rdkit.Chem import AllChem


HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}


class QM7(InMemoryDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    def atomref(self, target) -> Optional[torch.Tensor]:
        if target in atomrefs:
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[target])
            return out.view(-1, 1)
        return None

    @property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit  # noqa
            return ['gdb7.sdf', 'qm7.csv', 'uncharacterized.txt']
        except ImportError:
            return ['qm7_v3.pt']

    @property
    def processed_file_names(self) -> str:
        return 'data_v3.pt'

    def process(self):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None

        if rdkit is None:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)

            data_list = torch.load(self.raw_paths[0])
            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(self.collate(data_list), self.processed_paths[0])
            return

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        with open(self.raw_paths[1], 'r') as f:
            target = f.read().split('\n')[1:-1]
            smiles = [line.split(',')[0]
                      for line in target]
            target = [line.split(',')[1]
                      for line in target]
            
        #print(len(smiles))
        #print(self.raw_paths[0])
        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)
        #print(len(suppl))
        #mol1 = next(suppl)
        print(smiles[0])
        #print(suppl.GetItemText(6833))
        mol1=suppl[0]
        print(Chem.MolToSmiles(mol1))
        #p = list(mol1.GetConformer().GetAtomPosition(0))
        #print(p)
        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            #mol = Chem.MolFromSmiles(s)
            #mol = Chem.AddHs(mol)
            N = mol.GetNumAtoms()            
            pos = suppl.GetItemText(i).split('\n')[4:4 + N]
            pos = [[float(x) for x in line.split()[:3]] for line in pos]
            pos = torch.tensor(pos, dtype=torch.float)
            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

            z = torch.tensor(atomic_number, dtype=torch.long)
            
            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]
        
            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type,
                                  num_classes=len(bonds)).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N).tolist()

            x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                              dtype=torch.float).t().contiguous()
            x = torch.cat([x1.to(torch.float), x2], dim=-1)
            y = target[i] #.unsqueeze(0)
            s = smiles[i]
            m = Chem.MolFromSmiles(s)
            m = Chem.AddHs(m)
            # Get Atomic Position
            ## ETKDG ##
            AllChem.EmbedMultipleConfs(m, numConfs=3, randomSeed=2222, numThreads=0)
            rmslist = []
            AllChem.AlignMolConformers(m, RMSlist=rmslist)
            if not rmslist:
                continue
            rangeidx = len(rmslist)+1
            sortidx = [i for i in range(rangeidx)]
            minidx = sortidx[0]
            maxidx1, maxidx2 = sortidx[-1], sortidx[-2]   
            
            minpos = m.GetConformer(minidx).GetPositions()
            max1pos = m.GetConformer(maxidx1).GetPositions()
            max2pos = m.GetConformer(maxidx2).GetPositions()
            
            minpos_etkdg = torch.tensor(minpos, dtype=torch.float)
            max1pos_etkdg = torch.tensor(max1pos, dtype=torch.float)
            max2pos_etkdg = torch.tensor(max2pos, dtype=torch.float)
            
            data = Data(x=x, z=z, pos=pos, idx=i, smiles=s,
                        #minpos_etkdg=minpos_etkdg, max1pos_etkdg=max1pos_etkdg, max2pos_etkdg=max2pos_etkdg,
                        y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
        
    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict