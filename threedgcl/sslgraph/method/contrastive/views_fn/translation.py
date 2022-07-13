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
            max1pos_mmff = data.max1pos_mmff
            return Data(pos=max1pos_mmff, smiles=data.smiles, z=data.z)
        elif self.method == 'MMFF2':
            max2pos_mmff = data.max2pos_mmff
            return Data(pos=max2pos_mmff, smiles=data.smiles, z=data.z)
        elif self.method == 'MMFF3':
            max3pos_mmff = data.max3pos_mmff
            return Data(pos=max3pos_mmff, smiles=data.smiles, z=data.z)
        elif self.method == 'MMFF4':
            max4pos_mmff = data.max4pos_mmff
            return Data(pos=max4pos_mmff, smiles=data.smiles, z=data.z)

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

