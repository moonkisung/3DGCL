import copy
import gc
import torch
import numpy as np
import pandas as pd
import random

import torch.nn as nn
from torch.optim import Adam, SparseAdam
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import os
from tqdm import trange, tqdm

from dig.threedgraph.dataset.dataset import scaffold_split
from dig.threedgraph.dataset import MoleculeNet
from dig.threedgraph.dataset import QM9, QM
from dig.sslgraph.utils.seed import setup_seed
from dig.sslgraph.utils.cosine_annealing_with_warmup import CosineAnnealingWarmUpRestarts
#from dig.sslgraph.utils.parallel import DataParallelModel, DataParallelCriterion
from dig.sslgraph.utils import Encoder
from dig.sslgraph.utils.encoders import ShiftedSoftplus
from dig.threedgraph.method import SchNet

from torch_geometric.loader import DataLoader
#from torch_geometric.nn.acts import swish

from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch
from torch.utils import data as torch_data
import matplotlib.pyplot as plt


#criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()
mae_func = torch.nn.L1Loss()
mse_func = nn.MSELoss()

def train_cls(model, device, loader, optimizer):
    model.train()
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch).to(torch.float64)
        y = batch.y.view(pred.shape).to(torch.float64)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print('trainloss', loss.item())

def train_reg(model, device, loader, optimizer, mae=False, target='y'):
    model.train()
    for step, batch in enumerate(loader):   # , desc="Iteration"
        #torch.autograd.set_detect_anomaly(True)
        batch = batch.to(device)
        pred = model(batch)
        
        y = batch[target].view(pred.shape).to(torch.float64)
        #print('pred', pred)
        #print('y', y)
        if mae:
            loss = mae_func(pred, y)
        else:
            #print('rmse')
            loss = torch.sum((pred-y)**2)/y.size(0)
            #loss = torch.sqrt(mse_func(pred, y) + 1e-8)
            #loss = torch.sqrt(torch.mean((pred-y)**2))
        optimizer.zero_grad()
        #print(loss)
        loss.backward()
        optimizer.step()

def eval_cls(model, device, loader):
    model.eval()
    y_true = []
    y_scores = []
    cost = 0.0
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch).to(torch.float64)
            ## Loss ##
            y = batch.y.view(pred.shape).to(torch.float64)
            #print('y', y.view(-1))
            #print('pred', pred.view(-1))
            #print('sigmoid', torch.sigmoid(pred.view(-1)))
            loss = criterion(pred, y)
            cost += loss
        ## AUC ##    
        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)
    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
    roc_list = roc_auc_score(y_true, y_scores)
    cost = cost.cpu() / len(loader)
    return roc_list, cost


def eval_reg(model, device, loader, mae=False, target='y'):
    model.eval()
    y_true = []
    y_scores = []
    smiles = []
    
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
        y_true.append(batch[target].view(pred.shape))
        y_scores.append(pred)
        smiles +=batch.smiles
    
    y_true = torch.cat(y_true, dim = 0).cpu().numpy().flatten()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy().flatten()
    if mae:
        loss = mean_absolute_error(y_true, y_scores)
    else:
        loss = mean_squared_error(y_true, y_scores, squared=False)
    return loss, y_true, y_scores, smiles


class Finetune(object):
    
    def __init__(self, args, log_interval=1): #, **kwargs):  #reduction='sum'
        
        if args.dataset == 'esol':
            self.dataset = MoleculeNet(root='dataset/', name='esol')  # regression
            np.random.seed(args.seed)
            idx = np.random.choice(np.arange(1128), size=1128, replace=False)
            self.dataset = self.dataset[idx]
            #print(len(self.dataset))
            self.task_type = 'reg'
            self.mae = False
            self.out_dim = 1
        elif args.dataset == 'freesolv':
            self.dataset = MoleculeNet(root='dataset/', name='freesolv')  # regression
            self.task_type = 'reg'
            self.mae = False
            self.out_dim = 1
        elif args.dataset == 'lipo':
            self.dataset = MoleculeNet(root='dataset/', name='lipo')  # regression
            self.task_type = 'reg'
            self.mae = False
            self.out_dim = 1
        elif args.dataset == 'hiv':
            self.dataset = MoleculeNet(root='dataset/', name='hiv')  # binary
            self.task_type = 'cls'
            self.out_dim = 1
        elif args.dataset == 'bace':
            self.dataset = MoleculeNet(root='dataset/', name='bace')  # binary
            self.task_type = 'cls'
            self.out_dim = 1
        elif args.dataset == 'bbbp':
            self.dataset = MoleculeNet(root='dataset/', name='bbbp')  # binary
            self.task_type = 'cls'
            self.out_dim = 1
        elif args.dataset == 'clintox':
            self.dataset = MoleculeNet(root='dataset/', name='clintox')  # binary
            self.task_type = 'cls'
            self.out_dim = 1
        elif args.dataset == 'tox21':
            self.dataset = MoleculeNet(root='dataset/', name='tox21')  # multi
            self.task_type = 'cls'
            self.out_dim = 12
        elif args.dataset == 'sider':
            self.dataset = MoleculeNet(root='dataset/', name='sider')  # multi
            self.task_type = 'cls'
            self.out_dim = 27
        elif args.dataset == 'toxcast':
            self.dataset = MoleculeNet(root='dataset/', name='toxcast')  # multi
            self.task_type = 'cls'
            self.out_dim = 617
        elif args.dataset == 'qm7':
            self.dataset = QM(root='dataset/', name='qm7')  # regression
            self.task_type = 'reg'
            self.mae = True
            self.out_dim = 1
        elif args.dataset == 'qm8':
            self.dataset = QM(root='dataset/', name='qm8')  # regression
            #self.dataset.data.y = self.dataset.data.y.view(-1, 16)[: ,0]
            self.target = args.target
            self.dataset.data.y = self.dataset.data[self.target]
            self.task_type = 'reg'
            self.mae = True
            self.out_dim = 1
        elif args.dataset == 'qm9':
            self.dataset = QM9(root='dataset/qm9/')  # regression
            self.dataset.data.edge_index = None
            self.dataset.data.edge_attr = None
            self.dataset.data.y = self.dataset.data[args.target]
            self.task_type = 'qm9'
            self.out_dim = 1
        

        self.args = args
        self.seed  = args.seed
        self.device = args.device
        self.finetune = args.finetune
        self.encoder = args.encoder
        self.proj = args.proj
        self.model_path = args.model_path
        
        self.cutoff = args.cutoff
        self.num_layers = args.num_layers
        self.hidden_channels = args.z_dim
        if args.encoder=='schnet':
            self.num_filters = args.num_filters
            self.num_gaussians = args.num_gaussians
                            
        self.batch_size = args.batch_size
        self.n_times = args.n_times
        self.n_folds = args.n_folds
        
        self.f_epoch = args.f_epoch
        self.f_lr = args.f_lr
        self.z_dim = args.z_dim
        self.dropout_rate = args.dropout_rate
        self.f_optim = args.f_optim
        self.f_weight_decay = args.f_weight_decay
        self.f_lr_decay_step_size = args.f_lr_decay_step_size
        self.f_lr_decay_factor = args.f_lr_decay_factor
        
        self.expo_gamma = args.expo_gamma
        
        self.T_0 = args.T_0
        self.T_mult = args.T_mult
        self.eta_max = args.eta_max
        self.T_up = args.T_up
        self.gamma = args.gamma
        
        self.log_interval = 10
            
        #self.setup_train_config(batch_size, cutoff, num_layers, z_dim)
        # Use default config if not further specified
        

    def setup_train_config(self, batch_size, cutoff, num_layers, num_filters, num_gaussians, z_dim, dropout_rate, target, f_lr, f_weight_decay):
        self.batch_size = batch_size
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians
        self.z_dim = z_dim
        self.dropout_rate = dropout_rate
        self.target = target
        self.dataset.data.y = self.dataset.data[self.target]
        self.f_lr = f_lr
        self.f_weight_decay = f_weight_decay
        
        
    def evaluate(self, fold_seed=None):
        #setup_seed(self.seed)
        pred_head = self.proj
        if self.finetune:
            encoder = Encoder(self.args)
            encoder.load_state_dict(torch.load(self.model_path), strict=False)
            model = PredictionModel(encoder, self.encoder, self.num_layers, pred_head, self.z_dim, self.out_dim, self.dropout_rate)
            
        else: 
            if self.encoder == 'schnet':
                model = SchNet(cutoff=self.cutoff, num_layers=self.num_layers, hidden_channels=self.z_dim, out_channels=self.out_dim, num_filters=self.num_filters, num_gaussians=self.num_gaussians, dropout_rate=self.dropout_rate)
                
        num_params = sum(p.numel() for p in model.parameters())
        if self.task_type == 'cls':
            writer = SummaryWriter()
            train_aucs, val_aucs, test_aucs = [], [], [] # Fold 단위
            train_losses, val_losses, test_losses = [], [], [] # Fold 단위
            best_val_aucs, best_test_aucs = [], []
            best_val_losses, best_test_losses = [], []
            best_test_trues, best_test_preds, best_test_smiless = [], [], []
            
            for i in range(self.n_times):
                i+=1
                self.seed+=i
                for fold, train_loader, val_loader, test_loader in k_scaffold(self.n_folds, self.dataset,
                                                                              self.batch_size,self.task_type,self.seed):
                    fold_model = copy.deepcopy(model).to(self.device)
                    f_optimizer = Adam(fold_model.parameters(), lr=self.f_lr, weight_decay=self.f_weight_decay)
                    scheduler = ExponentialLR(f_optimizer, gamma=self.expo_gamma)
                    
                    wait = 0
                    best_val_loss = float("inf") 
                    best_test_loss = float("inf") 
                    patience = 15
                    fold_train_aucs, fold_val_aucs, fold_test_aucs = [], [], [] # Fold 단위
                    fold_train_losses, fold_val_losses, fold_test_losses = [], [], [] # Fold 단위
                    best_test_true, best_test_pred, best_test_smiles = [], [], []
                    
                    with trange(self.f_epoch) as t:
                        for epoch in t:
                            t.set_description('Finetune: epoch %d' % (epoch+1))
                            train_cls(fold_model, self.device, train_loader, f_optimizer)
                            train_auc, train_loss = eval_cls(fold_model, self.device, train_loader)
                            val_auc, val_loss = eval_cls(fold_model, self.device, val_loader)
                            test_auc, test_loss = eval_cls(fold_model, self.device, test_loader)
                            
                            train_aucs.append(train_auc)
                            val_aucs.append(val_auc)
                            test_aucs.append(test_auc)
                            
                            train_losses.append(train_loss)
                            val_losses.append(val_loss)
                            test_losses.append(test_loss)
                            
                            fold_train_aucs.append(train_auc)
                            fold_val_aucs.append(val_auc)
                            fold_test_aucs.append(test_auc)
                            
                            fold_train_losses.append(train_loss)
                            fold_val_losses.append(val_loss)
                            fold_test_losses.append(test_loss)
                            
                            scheduler.step()
                            if np.less(val_loss, best_val_loss):
                                best_val_loss = val_loss
                                best_test_loss = test_loss
                                best_val_auc = val_auc
                                best_test_auc = test_auc
                                
                                wait = 0
                            else:
                                wait += 1
                                if wait >= patience:
                                    print('Early stop at Epoch: {:d} with final val loss: {:.4f}'.format(epoch+1, val_auc))
                                    break
                            
                            t.set_postfix(best_val_loss='{:.3f}'.format(float(best_val_loss)), 
                                          best_test_loss='{:.3f}'.format(float(best_test_loss)),
                                          best_val_auc='{:.3f}'.format(float(best_val_auc)), 
                                          best_test_auc='{:.3f}'.format(float(best_test_auc)),
                                          
                                          train_loss='{:.3f}'.format(float(train_loss)), 
                                          val_loss='{:.3f}'.format(float(val_loss)), 
                                          test_loss='{:.3f}'.format(float(test_loss)),
                                          
                                          train_auc='{:.3f}'.format(float(train_auc)), 
                                          val_auc='{:.3f}'.format(float(val_auc)), 
                                          test_auc='{:.3f}'.format(float(test_auc)),
                                          FOLD='{:.1f}'.format(fold))
                                                   
                        best_val_aucs.append(best_val_auc)
                        best_test_aucs.append(best_test_auc)
                        best_val_losses.append(best_val_loss)
                        best_test_losses.append(best_test_loss)
            
            auc = np.mean(best_test_aucs)
            auc_sd = np.std(best_test_aucs)
            loss = np.mean(best_test_losses)
            sd = np.std(best_test_losses)
            return auc, auc_sd, loss, sd
        
                
        elif self.task_type == 'reg':
            writer = SummaryWriter()
            # log_dir='./results/' + self.args.dataset + '/' + self.encoder
            train_losses, val_losses, test_losses = [], [], [] # Fold 단위
            best_val_losses, best_test_losses = [], []
            best_test_trues, best_test_preds, best_test_smiless = [], [], []
            for i in range(self.n_times):
                i+=1
                self.seed+=i
                for fold, train_loader, val_loader, test_loader in k_scaffold(self.n_folds, self.dataset,
                                                                              self.batch_size,self.task_type, self.seed):
                    fold_model = copy.deepcopy(model).to(self.device)
                    

                    if self.f_optim == 'StepLR':
                        f_optimizer = Adam(fold_model.parameters(), lr=self.f_lr, weight_decay=self.f_weight_decay)
                        scheduler = StepLR(f_optimizer, step_size=self.f_lr_decay_step_size, gamma=self.f_lr_decay_factor)
                    elif self.f_optim == 'ExponentialLR':
                        f_optimizer = Adam(fold_model.parameters(), lr=self.f_lr, weight_decay=self.f_weight_decay)
                        scheduler = ExponentialLR(f_optimizer, gamma=self.expo_gamma)    


                    wait = 0
                    best_val_loss = float("inf") 
                    best_test_loss = float("inf") 
                    patience = 30
                    fold_train_losses, fold_val_losses, fold_test_losses = [], [], [] # Fold 단위
                    best_test_true, best_test_pred, best_test_smiles = [], [], []
                    with trange(self.f_epoch) as t:
                        for epoch in t:
                            t.set_description('Finetune: epoch %d' % (epoch+1))
                            train_reg(fold_model, self.device, train_loader, f_optimizer, mae=self.mae, target=self.target)
                            train_rmse, _, _, _ = eval_reg(fold_model, self.device, train_loader, mae=self.mae, target=self.target)
                            val_rmse, _, _, _ = eval_reg(fold_model, self.device, val_loader, mae=self.mae, target=self.target)
                            test_rmse, test_true, test_pred, test_smiles=eval_reg(fold_model, self.device, test_loader, mae=self.mae, target=self.target)
                            train_losses.append(train_rmse)
                            val_losses.append(val_rmse)
                            test_losses.append(test_rmse)
                            fold_train_losses.append(train_rmse)
                            fold_val_losses.append(val_rmse)
                            fold_test_losses.append(test_rmse)
                            scheduler.step()
                            
                            if np.less(val_rmse, best_val_loss):
                                best_val_loss = val_rmse
                                best_test_loss = test_rmse
                                best_test_true = test_true
                                best_test_pred = test_pred
                                best_test_smiles = test_smiles
                                wait = 0
                            else:
                                wait += 1
                                if wait >= patience:
                                    print('Early stop at Epoch: {:d} with final val loss: {:.4f}'.format(epoch+1, val_rmse))
                                    break
                            t.set_postfix(best_val_loss='{:.3f}'.format(float(best_val_loss)), 
                                          best_test_loss='{:.3f}'.format(float(best_test_loss)),
                                          train_rmse='{:.3f}'.format(float(train_rmse)), 
                                          val_rmse='{:.3f}'.format(float(val_rmse)), 
                                          test_rmse='{:.3f}'.format(float(test_rmse)),
                                          FOLD='{:.1f}'.format(fold))
                        train_sizes = [i + 1 for i in range(len(fold_train_losses))]
                        best_val_losses.append(best_val_loss)
                        best_test_losses.append(best_test_loss)
                        best_test_trues.append(best_test_true)
                        best_test_preds.append(best_test_pred)
                        best_test_smiless.append(best_test_smiles)
                       
            loss = np.mean(best_test_losses)
            sd = np.std(best_test_losses)
            return loss, sd, best_test_trues, best_test_preds, best_test_smiless
        

    
    def grid_search(self, args):
        finetune = args.finetune
        model = args.model_path.split('/')[-1]
        parameter = args.model_path.split('/')[2]
        n_times = args.n_times
        n_folds = args.n_folds
        batch_lst = args.batch_lst
        cutoff_lst = args.cutoff_lst
        num_layers_lst = args.num_layers_lst
        num_filters_lst= args.num_filters_lst
        num_gaussians_lst= args.num_gaussians_lst
        z_dim_lst = args.z_dim_lst
        dropout_rate_lst = args.dropout_rate_lst
        target_lst = args.target_lst
        f_lr_lst = args.f_lr_lst
        f_weight_decay_lst = args.f_weight_decay_lst
        
        directory = './results/' + self.args.dataset + '/' + self.encoder
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' +  directory)
        txtfile = directory + '/' + 'results.txt'
        
        if self.task_type=='cls':
            auc_m_lst = []
            auc_sd_lst = []
            loss_m_lst = []
            loss_sd_lst = []
            paras = []
            for batch_size in batch_lst:
                for cutoff in cutoff_lst:
                    for num_layers in num_layers_lst:
                        for num_filters in num_filters_lst:
                            for num_gaussians in num_gaussians_lst:
                                for z_dim in z_dim_lst:
                                    for dropout_rate in dropout_rate_lst:
                                        for target in target_lst:
                                            for f_lr in f_lr_lst:
                                                for f_weight_decay in f_weight_decay_lst:
                                                    self.setup_train_config(batch_size=batch_size, cutoff=cutoff,
                                                                            num_layers=num_layers, num_filters=num_filters, num_gaussians=num_gaussians,
                                                                            z_dim=z_dim, dropout_rate=dropout_rate, target=target, f_lr=f_lr, 
                                                                            f_weight_decay=f_weight_decay)
                                                    auc_m, auc_sd, loss_m, loss_sd = self.evaluate()
                                                    auc_m_lst.append(auc_m)
                                                    auc_sd_lst.append(auc_sd)
                                                    loss_m_lst.append(loss_m)
                                                    loss_sd_lst.append(loss_sd)
                                                    paras.append((batch_size, cutoff, num_layers, z_dim, dropout_rate, f_lr, f_weight_decay))
                                                    if self.f_optim == 'ExponentialLR':
                                                        with open(txtfile, "a") as myfile:
                                                            if finetune:
                                                                myfile.write('model:'+ str(model) + '_' + str(parameter) + '_')
                                                            myfile.write('finetune:'+ str(finetune) + '_n_times:' + str(n_times) +  '_folds:' + str(n_folds) + '_auc:'+ str(np.round(auc_m, 4)) + '_auc_sd:'+ str(np.round(auc_sd, 4)) + '_loss:'+ str(np.round(loss_m, 4)) + '_sd:'+ str(np.round(loss_sd, 4)) + '_batch:' + str(batch_size) + '_cutoff:' + str(cutoff) + '_layer:' + str(num_layers) + '_filter:' + str(self.num_filters) + '_gau:' + str(self.num_gaussians) + '_z_dim:' + str(z_dim) + '_dropout:' + str(dropout_rate) + '_lr:' + str(f_lr) + '_weight_decay:' + str(self.f_weight_decay) + '_expo_gamma:' +  str(self.expo_gamma) + "\n")    
                                        
            idx = np.argmin(loss_m_lst)
            
            total = [(param, auc) for param, auc in zip(paras, auc_m_lst)]
            print('Total Results: ', total)
            
            return auc_m_lst[idx], auc_sd_lst[idx], loss_m_lst[idx], loss_sd_lst[idx], paras[idx], total, self.args

        
        elif self.task_type=='reg':
            loss_m_lst = []
            loss_sd_lst = []
            paras = []
            list_test_trues, list_test_preds, list_test_smiles = [], [], []
            for batch_size in batch_lst:
                gc.collect()
                torch.cuda.empty_cache()
                for cutoff in cutoff_lst:
                    for num_layers in num_layers_lst:
                        for num_filters in num_filters_lst:
                            for num_gaussians in num_gaussians_lst:
                                for z_dim in z_dim_lst:
                                    for dropout_rate in dropout_rate_lst:
                                        for target in target_lst:
                                            for f_lr in f_lr_lst:
                                                for f_weight_decay in f_weight_decay_lst:
                                                    self.setup_train_config(batch_size=batch_size, cutoff=cutoff,
                                                                            num_layers=num_layers, num_filters=num_filters, num_gaussians=num_gaussians,
                                                                            z_dim=z_dim, dropout_rate=dropout_rate, target=target, f_lr=f_lr, 
                                                                            f_weight_decay=f_weight_decay)
                                                    loss_m, loss_sd, test_trues, test_preds, test_smiles = self.evaluate()
                                                    loss_m_lst.append(loss_m)
                                                    loss_sd_lst.append(loss_sd)
                                                    list_test_trues.append(test_trues)
                                                    list_test_preds.append(test_preds)
                                                    list_test_smiles.append(test_smiles)
                                                    #dic = {'smiles':test_smiles, 'y_true':test_trues, 'y_pred':test_preds}
                                                    #df = pd.DataFrame(dic)
                                                    #df['diff'] = abs(df['y_true'] - df['y_pred'])
                                                    #directory = './results/' + self.args.dataset + '/' + self.encoder
                                                    #csv = directory + '/' + f'{dropout_rate}_{f_weight_decay}.csv'
                                                    #df.to_csv(csv, index=False)
                                                    
                                                    paras.append((batch_size, cutoff, num_layers, z_dim, dropout_rate))
                                                    if self.f_optim == 'StepLR':
                                                        with open(txtfile, "a") as myfile:
                                                            if finetune:
                                                                myfile.write('model:'+ str(model) + '_' + str(parameter) + '_')
                                                            myfile.write('finetune:'+ str(finetune) + '_n_times:' + str(n_times) + '_folds:' + str(n_folds) + '_target:' + str(target) + '_loss:'+ str(np.round(loss_m, 5)) + '_sd:'+ str(np.round(loss_sd, 5)) + '_batch:' + str(batch_size) + '_cutoff:' + str(cutoff) + '_layer:' + str(num_layers) + '_filter:' + str(self.num_filters) + '_gau:' + str(self.num_gaussians) + '_z_dim:' + str(z_dim) + '_dropout:' + str(dropout_rate) + '_lr:' + str(f_lr) + '_weight_decay:' + str(self.f_weight_decay) + '_lr_decay_step_size:' +  str(self.f_lr_decay_step_size) + '_lr_decay_factor:' + str(self.f_lr_decay_factor) + "\n")
                                                    elif self.f_optim == 'ExponentialLR':
                                                        with open(txtfile, "a") as myfile:
                                                            if finetune:
                                                                myfile.write('model:'+ str(model) + '_' + str(parameter) + '_')
                                                            myfile.write('finetune:'+ str(finetune) + '_n_times:' + str(n_times) +  '_folds:' + str(n_folds) + '_target:' + str(target) + '_loss:'+ str(np.round(loss_m, 5)) + '_sd:'+ str(np.round(loss_sd, 5)) + '_batch:' + str(batch_size) + '_cutoff:' + str(cutoff) + '_layer:' + str(num_layers) + '_filter:' + str(self.num_filters) + '_gau:' + str(self.num_gaussians) + '_z_dim:' + str(z_dim) + '_dropout:' + str(dropout_rate) + '_lr:' + str(f_lr) + '_weight_decay:' + str(self.f_weight_decay) + '_expo_gamma:' +  str(self.expo_gamma) + "\n")    
                                        
            idx = np.argmin(loss_m_lst)
            total = [(param, loss) for param, loss in zip(paras, loss_m_lst)]
            print('Total Results: ', total)
            self.args.list_test_trues = list_test_trues
            self.args.list_test_preds = list_test_preds
            self.args.list_test_smiles = list_test_smiles
            
            return loss_m_lst[idx], loss_sd_lst[idx], paras[idx], total, self.args

    
    def get_embed(self, model, loader):
    
        model.eval()
        ret, y = [], []
        with torch.no_grad():
            for data in loader:
                y.append(data.y.numpy())
                #data.to(self.device)
                data = data.cuda()
                embed = model(data)
                ret.append(embed.cpu().numpy())

        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y
                
    
    def get_optim(self, optim):
        
        if callable(optim):
            return optim
        
        optims = {'Adam': torch.optim.Adam}
        
        return optims[optim]
    

class PredictionModel(nn.Module):
    
    def __init__(self, encoder, name, num_layers, pred_head, dim, out_dim, dropout):
        
        super(PredictionModel, self).__init__()
        self.encoder = encoder
        
        if pred_head == 'schnet':
            self.pred_head = nn.Sequential(nn.Linear(dim, dim//2),
                                 ShiftedSoftplus(),
                                 nn.Dropout(dropout),
                                 nn.Linear(dim//2, out_dim))
            self.sigmoid = nn.Sigmoid()
            
        elif pred_head == 'spherenet':
            init_z_dim = num_layers*dim if 'gin' in name or 'gcn' in name else dim
            self.pred_head = nn.Sequential(nn.Linear(init_z_dim, dim//2),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(dropout),
                                 nn.Linear(dim//2, out_dim))
            #self.pred_head = nn.Linear(dim, out_dim)
        
    def forward(self, data):
        zg = self.encoder(data)
        out = self.pred_head(zg)
        #out = self.sigmoid(out)
        return out
                      
                      
def checkbinary(yslist):
    check = []
    for ys in yslist:
        if 1 & 0 in ys:  # pass
            check.append(1)
        else:            # fold 한번 더
            check.append(2)
    return check    


def k_scaffold(n_folds, dataset, batch_size, task_type, seed):
    
    i = 0
    seed = seed
    seed = random.randint(0, 1e8)
    while i < n_folds:
        train, val, test = scaffold_split(dataset, seed+i)
        if len(val) <= 25 or len(test) <= 25:
            seed -= random.randint(0, 22)
            seed += 1
            continue
            
        train_loader = DataLoader(train, batch_size, shuffle=True)
        val_loader = DataLoader(val, batch_size, shuffle=False)
        test_loader = DataLoader(test, batch_size, shuffle=False)
                        
        train_ys = [ data.y for idx, data in enumerate(train_loader) ]
        val_ys = [ data.y for idx, data in enumerate(val_loader) ]
        test_ys = [ data.y for idx, data in enumerate(test_loader) ]
        if task_type == 'cls':
            print(len(train), len(val), len(test))
            checklist = checkbinary(train_ys) + checkbinary(val_ys) + checkbinary(test_ys)
            if 2 in checklist :
                seed -= random.randint(0, 22)
                seed += 1
                continue
            else:
                i+=1
            yield i, train_loader, val_loader, test_loader
        elif task_type == 'reg':
            print(len(train), len(val), len(test))
            if len(val)!=0 or len(test)!=0:
                i+=1
                #seed -= random.randint(0, 100)
                seed += 1
                #print('len(train), len(val), len(test)', len(train), len(val), len(test))
            yield i, train_loader, val_loader, test_loader
            
            
def k_fold(n_folds, dataset, batch_size, task_type):  # , seed=12345
    
    kf = KFold(n_folds, shuffle=True) # , random_state=seed

    test_indices, train_indices = [], []
    for _, idx in kf.split(torch.zeros(len(dataset))):
        test_indices.append(torch.from_numpy(idx))
        
    val_indices = [test_indices[i - 1] for i in range(n_folds)]
    
    for i in range(n_folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        idx_train = train_mask.nonzero(as_tuple=False).view(-1)
        train_indices.append(idx_train)
    for i in range(n_folds):
        if batch_size is None:
            batch_size = len()
        train_loader = DataLoader(dataset[train_indices[i].long()], batch_size, shuffle=True)
        val_loader = DataLoader(dataset[val_indices[i].long()], batch_size, shuffle=False)
        test_loader = DataLoader(dataset[test_indices[i].long()], batch_size, shuffle=False)
        print(len(train_indices[0]))
        print(len(val_indices[0]))
        print(len(test_indices[0]))
        yield i, train_loader, val_loader, test_loader
