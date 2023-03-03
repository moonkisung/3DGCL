import copy
import torch
import numpy as np
import gc
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, ExponentialLR

from tqdm import trange, tqdm
from dig.threedgraph.dataset.dataset import scaffold_split
from dig.threedgraph.dataset import MoleculeNet, QM
from dig.sslgraph.utils.seed import setup_seed
from dig.sslgraph.utils.cosine_annealing_with_warmup import CosineAnnealingWarmUpRestarts

from torch_geometric.loader import DataLoader

from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

import torch
from torch.utils import data as torch_data


class Pretrain(object):
    r"""   
    Args:
        log_interval (int, optional): Perform evaluation per k epochs. (default: :obj:`1`)
        epoch_select (string, optional): :obj:`"test_max"` or :obj:`"val_max"`.
            (default: :obj:`"test_max"`)
        **kwargs (optional): Training and evaluation configs in :meth:`setup_train_config`.
        
    Examples
    --------
    >>> encoder = Encoder(...)
    >>> model = Contrastive(...)
    >>> evaluator = GraphUnsupervised(dataset, log_interval=10, device=0, p_lr = 0.001)
    >>> evaluator.evaluate(model, encoder)
    """
    
    def __init__(self, args, log_interval=10, epoch_select='val', 
                  device=None, **kwargs):  #reduction='sum'

        if args.pretrain_dataset == 'pubchem01':
            self.pretrain_dataset = MoleculeNet(root='dataset/', name='pubchem01')  # regression
        elif args.pretrain_dataset == 'moleculenet':
            self.pretrain_dataset = MoleculeNet(root='dataset/', name='moleculenet')  # regression
        elif args.pretrain_dataset == 'esol':
            self.pretrain_dataset = MoleculeNet(root='dataset/', name='esol')  # regression
        elif args.pretrain_dataset == 'freesolv':
            self.pretrain_dataset = MoleculeNet(root='dataset/', name='freesolv')  # regression
        elif args.pretrain_dataset == 'lipo':
            self.pretrain_dataset = MoleculeNet(root='dataset/', name='lipo')  # regression
        elif args.pretrain_dataset == 'hiv':
            self.pretrain_dataset = MoleculeNet(root='dataset/', name='hiv')  # binary
        elif args.pretrain_dataset == 'bace':
            self.pretrain_dataset = MoleculeNet(root='dataset/', name='bace')  # binary
        elif args.pretrain_dataset == 'bbbp':
            self.pretrain_dataset = MoleculeNet(root='dataset/', name='bbbp')  # binary
        elif args.pretrain_dataset == 'clintox':
            self.pretrain_dataset = MoleculeNet(root='dataset/', name='clintox')  # binary
        elif args.pretrain_dataset == 'tox21':
            self.pretrain_dataset = MoleculeNet(root='dataset/', name='tox21')  # multi
        elif args.pretrain_dataset == 'sider':
            self.pretrain_dataset = MoleculeNet(root='dataset/', name='sider')  # multi
        elif args.pretrain_dataset == 'toxcast':
            self.pretrain_dataset = MoleculeNet(root='dataset/', name='toxcast')  # multi
        elif args.pretrain_dataset == 'qm7':
            self.pretrain_dataset = QM(root='dataset/', name='qm7')  # regression
        elif args.pretrain_dataset == 'qm8':
            self.pretrain_dataset = QM(root='dataset/', name='qm8')  # regression
        
        self.seed  = args.seed
        self.device = args.device
        self.finetune = args.finetune
        
        self.batch_size = args.batch_size
        self.p_epoch = args.p_epoch
        self.p_lr = args.p_lr
        self.p_optim = args.p_optim
        self.p_weight_decay = args.p_weight_decay
        self.z_dim = args.z_dim
        self.dropout = args.dropout_rate
        
        self.p_weight_decay = args.p_weight_decay
        self.p_lr_decay_step_size = args.p_lr_decay_step_size
        self.p_lr_decay_factor = args.p_lr_decay_factor
        
        self.expo_gamma = args.expo_gamma
        
        self.T_0 = args.T_0
        self.T_mult = args.T_mult
        self.eta_max = args.eta_max
        self.T_up = args.T_up
        self.gamma = args.gamma
        
        self.epoch_select = epoch_select
        self.log_interval = log_interval
                
        
    def evaluate(self, learning_model, encoder, fold_seed=None):
        pretrain_loader = DataLoader(self.pretrain_dataset, self.batch_size, shuffle=True)
        gc.collect()
        torch.cuda.empty_cache()
        if self.p_optim == 'StepLR':
            p_optimizer = Adam(encoder.parameters(), lr=self.p_lr, weight_decay=self.p_weight_decay)
            p_scheduler = StepLR(p_optimizer, step_size=self.p_lr_decay_step_size, gamma=self.p_lr_decay_factor)
        elif self.p_optim == 'ExponentialLR':
            p_optimizer = Adam(encoder.parameters(), lr=self.p_lr, weight_decay=self.p_weight_decay)
            p_scheduler = ExponentialLR(p_optimizer, gamma=self.expo_gamma)
        elif self.p_optim == 'Cosine':
            p_optimizer = Adam(encoder.parameters(), lr=0)
            p_scheduler = CosineAnnealingWarmUpRestarts(p_optimizer, 
                                                        T_0=self.T_0, T_mult=self.T_mult, eta_max=self.eta_max, T_up=self.T_up, gamma=self.gamma)
        #p_optimizer = self.get_optim(self.p_optim)(encoder.parameters(), lr=self.p_lr, weight_decay=self.p_weight_decay)
        encoder = next(learning_model.train(encoder, pretrain_loader, p_optimizer, p_scheduler, self.p_epoch), None)
        gc.collect()
        torch.cuda.empty_cache()
        return encoder               