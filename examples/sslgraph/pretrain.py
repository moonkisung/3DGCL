from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))

import pandas as pd
import matplotlib.pyplot as plt
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')
import torch
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../..')

from dig.sslgraph.utils import Encoder
from dig.sslgraph.evaluation import Pretrain
from dig.threedgraph.dataset import MoleculeNet, QM
from dig.sslgraph.method import GraphCL

import argparse

paser = argparse.ArgumentParser()
args = paser.parse_args("")

def main():
    # Finetune or rand init
    args.finetune = False
    args.seed = 2222

    # File Path
    args.model_path = './models'

    # Device
    args.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # Dataset
    args.pretrain_dataset = 'esol'
    args.batch_size = 400
    
    # Model
    args.encoder = 'schnet'
    args.cutoff = 5.0  # [5.0, 10.0]
    args.num_layers = 2 # [2, 4]
    args.num_filters = 128
    args.num_gaussians = 50
    args.z_dim = 512

    args.int_emb_size = 64
    args.basis_emb_size_dist = 8
    args.basis_emb_size_angle = 8
    args.basis_emb_size_torsion = 8
    args.out_emb_channels = 256
    args.num_spherical = 3
    args.num_radial = 6
    args.envelope_exponent = 5
    args.num_before_skip = 1
    args.num_after_skip = 2
    args.num_output_layers = 3
    args.use_node_features = True
    
    # Learning
    args.p_epoch = 300
    args.p_lr = 1e-3
    args.aug_1, args.aug_2 = 'MMFFrandom', 'MMFFrandom'
    args.aug_ratio = 0.25
    args.tau = 0.2
    args.proj = 'spherenet'

    # Regularization
    args.dropout_rate = 0.0

    args.p_optim = 'ExponentialLR' #['StepLR', ExponentialLR, 'Cosine']
    args.p_weight_decay = 0
    
    #'StepLR'
    args.p_lr_decay_step_size = 15  # 15 epoch 마다 lr * p_lr_decay_factor
    args.p_lr_decay_factor = 0.5

    # ExponentialLR
    args.expo_gamma = 0.95
    
    # Cosine
    args.T_0 = 20        # 최초 주기값
    args.T_mult = 2      # 최초 주기값에 비해 얼만큼 주기를 늘려갈 것인지
    args.eta_max = 0.05  # lr 최대값
    args.T_up = 10      # Warm up 시 필요한 epoch 수(일반적으로 짧은 수)
    args.gamma = 0.5     # 주기가 반복될수록 곱해지는 scale 값

    encoder = Encoder(args)
    graphcl = GraphCL(args)  # , device=args.device
    evaluator = Pretrain(args)
    encoder = evaluator.evaluate(learning_model=graphcl, encoder=encoder)
    
if __name__ == "__main__":
    main()
