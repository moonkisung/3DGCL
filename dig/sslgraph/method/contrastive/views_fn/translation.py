import re

from sklearn.utils import shuffle
import random
import torch
import torch_geometric.transforms as T
import numpy as np
from torch_geometric.data import Batch, Data
import rdkit.Chem.AllChem as AllChem
from rdkit import Chem



class NodeTranslation():
    '''Node attribute masking on the given graph or batched graphs. 
    Class objects callable via method :meth:`views_fn`.
    
    Args:
        mode (string, optinal): Masking mode with three options:
            :obj:`"whole"`: mask all feature dimensions of the selected node with a Gaussian distribution;
            :obj:`"partial"`: mask only selected feature dimensions with a Gaussian distribution;
            :obj:`"onehot"`: mask all feature dimensions of the selected node with a one-hot vector.
            (default: :obj:`"whole"`)
        mask_ratio (float, optinal): The ratio of node attributes to be masked. (default: :obj:`0.1`)
        mask_mean (float, optional): Mean of the Gaussian distribution to generate masking values.
            (default: :obj:`0.5`)
        mask_std (float, optional): Standard deviation of the distribution to generate masking values. 
            Must be non-negative. (default: :obj:`0.5`)
    '''
    def __init__(self, method, device, std=0.01):  # method
        self.method = method
        self.std = std
        self.device = device
    
    def __call__(self, data):
        return self.views_fn(data)
    def do_trans(self, data):
        if self.method == 'MMFF1':
            #max1pos_mmff = data.max1pos_mmff
            return Data(pos=data.max1pos_mmff, smiles=data.smiles, z=data.z, energy=data.max1_energy, min_energy=data.min_energy)
        elif self.method == 'MMFF2':
            return Data(pos=data.max2pos_mmff, smiles=data.smiles, z=data.z, energy=data.max2_energy, min_energy=data.min_energy)
        elif self.method == 'MMFF3':
            return Data(pos=data.max3pos_mmff, smiles=data.smiles, z=data.z, energy=data.max3_energy, min_energy=data.min_energy)
        elif self.method == 'MMFF4':
            return Data(pos=data.max4pos_mmff, smiles=data.smiles, z=data.z, energy=data.max4_energy, min_energy=data.min_energy)
        elif self.method == 'UFF1':
            max1pos_uff = data.max1pos_uff
            return Data(pos=max1pos_uff, smiles=data.smiles, z=data.z)
        elif self.method == 'UFF2':
            max2pos_uff = data.max2pos_uff
            return Data(pos=max2pos_uff, smiles=data.smiles, z=data.z)
        elif self.method == 'rotation':
            axis = torch.randint(0,3, (1,))  # 0~2 사이의 axis random하게 생성
            rotation_axis = T.RandomRotate(degrees=180, axis=axis)
            rotation = rotation_axis(data)
            return Data(pos=rotation.pos, smiles=data.smiles, z=data.z)
        elif self.method == 'noise':  # Gaussian noise (sampled from normal distribution with mean 0 and variance 0.01)
            # https://stackoverflow.com/questions/59090533/how-do-i-add-some-gaussian-noise-to-a-tensor-in-pytorch
            noise = torch.randn(data.pos.shape).to(self.device) * 0.01   #* (self.std**0.5)
            noise = data.pos + noise
            return Data(pos=noise, smiles=data.smiles, z=data.z)
        #smiles = data.smiles
        #pos = data.pos.detach().clone()
        #print('first pos: ', pos)
        #mol = Chem.MolFromSmiles(smiles)
        #mol = Chem.AddHs(mol)
        #n_atoms = len(mol.GetAtoms()) # 분자의 원자 개수
        # Get Atomic Position
        #for i in range(self.repeat):
        #    status = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        #    if status !=0:
        #        print(f"Error while generating 3D: {Chem.MolToSmiles(mol)}")
        #        continue
            
        #pos_list = []
        #for atm_id in range(n_atoms):
        #    atm_pos = mol.GetConformer(0).GetAtomPosition(atm_id)
        #    crd = [atm_pos.x, atm_pos.y, atm_pos.z]
        #    pos_list.append(crd)
        #pos = torch.tensor(pos_list, dtype=torch.float)
        #print('last pos: ', pos)
        #return Data(pos=pos, smiles=data.smiles, z=data.z)
        #return Data(pos=pos, smiles=data.smiles, z=data.z)

    def views_fn(self, data):
        #data = data.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        r"""Method to be called when :class:`NodeAttrMask` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """
        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return self.do_trans(data)

