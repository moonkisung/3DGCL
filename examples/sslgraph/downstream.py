import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../..')
import torch

from threedgcl.sslgraph.utils import Encoder
from threedgcl.sslgraph.evaluation import GraphUnsupervised
from threedgcl.sslgraph.evaluation import Finetune
from threedgcl.threedgraph.dataset import MoleculeNet
from threedgcl.threedgraph.method import SphereNet, SchNet, DimeNetPP
from threedgcl.sslgraph.method import GraphCL

import matplotlib.pyplot as plt

from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')

import pandas as pd
from rdkit import Chem
import rdkit.Chem.AllChem as AllChem

import argparse

paser = argparse.ArgumentParser()
args = paser.parse_args("")

### Finetune or rand init
args.finetune = False
args.seed = 2222

# File Path
args.model_path = './models/batch-400_proj-spherenet_cutoff-5.0_layers-2_filter-128_gau-50_z_dim-512_lr-0.001_\
aug_1-noise_aug_2-noise_aug_ratio-0.25_tau-0.2_optim-ExponentialLR_weight_decay-0_expo_gamma-0.95_dropout-0.0/enc_best_epoch-276_loss-12.075.pkl'

# Device
args.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Dataset
args.dataset = 'esol'
args.batch_size = 32

# Model
args.encoder = 'schnet'
args.cutoff = 5.0  # [5.0, 10.0]
args.num_layers = 2 # [2, 4]
args.num_filters = 128
args.num_gaussians = 50
args.z_dim = 512

# Learning
args.n_times = 3
args.n_folds = 3
args.f_epoch = 200
args.f_lr = 1e-3
args.aug_1, args.aug_2 = 'ETKDG1', 'ETKDG2'
args.aug_ratio = 0.2
args.tau = 0.2
args.proj = 'schnet'

# Regularization
args.dropout_rate = 0.0

args.f_optim = 'ExponentialLR' #['StepLR', ExponentialLR, 'Cosine']
args.f_weight_decay = 5e-5

#'StepLR'
args.f_lr_decay_step_size = 20  # 15 epoch 마다 lr * p_lr_decay_factor
args.f_lr_decay_factor = 0.5

# ExponentialLR 
args.expo_gamma = 0.97

# Cosine
args.T_0 = 100        # 최초 주기값
args.T_mult = 1      # 최초 주기값에 비해 얼만큼 주기를 늘려갈 것인지
args.eta_max = 0.05  # lr 최대값
args.T_up = 10        # Warm up 시 필요한 epoch 수(일반적으로 짧은 수)
args.gamma = 0.5     # 주기가 반복될수록 곱해지는 scale 값

args.batch_lst = [32]
args.cutoff_lst = [5.0]
args.num_layers_lst = [2]
args.num_filters_lst = [128]
args.num_gaussians_lst = [50]
args.z_dim_lst = [512]
args.dropout_rate_lst = [0.3, 0.0, 0.1, 0.2, 0.25, 0.3, 0.35]
args.target_lst = ['y']
args.f_lr_lst = [1e-3]
args.f_weight_decay_lst = [0, 5e-5, 5e-4, 1e-3, 5e-5, 1e-4, 1e-5]

evaluator = Finetune(args=args, log_interval=10)
auc_m_lst, auc_sd_lst, paras, total = evaluator.grid_search(args)
#loss, sd = evaluator.evaluate()
