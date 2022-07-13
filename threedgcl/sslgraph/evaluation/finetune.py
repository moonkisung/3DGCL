import copy
import torch
import numpy as np
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
from dig.threedgraph.method import SphereNet, SchNet, DimeNetPP

from torch_geometric.loader import DataLoader
from torch_geometric.nn.acts import swish

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


criterion = nn.BCEWithLogitsLoss(reduction = "none")
mae_func = torch.nn.L1Loss()
mse_func = nn.MSELoss()

def train_cls(model, device, loader, optimizer):
    model.train()
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        #batch = batch.cuda()
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        #loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat).to(loss_mat.dtype))

        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()
        optimizer.step()
        #print('trainloss', loss.item())


def train_reg(model, device, loader, optimizer, mae=False, target='y'):
    model.train()
    for step, batch in enumerate(loader):   # , desc="Iteration"
        #torch.autograd.set_detect_anomaly(True)
        batch = batch.to(device)
        pred = model(batch)
        #print(batch)
        y = batch[target].view(pred.shape).to(torch.float64)
        #print('pred', pred)
        #print('y', y)
        if mae:
            #loss = torch.sum(abs(pred-y))/y.size(0)
            #print('mae')
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

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        #batch = batch.cuda()
        with torch.no_grad():
            pred = model(batch)
            
            ## Loss ##
            y = batch.y.view(pred.shape).to(torch.float64)
            #Whether y is non-null or not.
            is_valid = y**2 > 0
            #Loss matrix
            loss_mat = criterion(pred.double(), (y+1)/2)
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat)/torch.sum(is_valid)
            
        ## AUC ##    
        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)
    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))
        #loss_list.append(log_loss(y_true.cpu(), y_scores.cpu(), eps=1e-6))
    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))
    
    loss = loss.item()
    return sum(roc_list)/len(roc_list), loss


def eval_reg(model, device, loader, mae=False, target='y'):
    model.eval()
    y_true = []
    y_scores = []
    
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
        y_true.append(batch[target].view(pred.shape))
        y_scores.append(pred)
    
    y_true = torch.cat(y_true, dim = 0).cpu().numpy().flatten()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy().flatten()
    if mae:
        #print('y_true', y_true)
        #print('y_scores', y_scores)
        loss = mean_absolute_error(y_true, y_scores)
    else:
        loss = mean_squared_error(y_true, y_scores, squared=False)
    #cor = pearsonr(y_true, y_scores)[0]
    return loss


class Finetune(object):
    r"""   
    Args:
        dataset (torch_geometric.data.Dataset): The graph classification dataset.
        classifier (string, optional): Linear classifier for evaluation, :obj:`"SVC"` or 
            :obj:`"LogReg"`. (default: :obj:`"SVC"`)
        log_interval (int, optional): Perform evaluation per k epochs. (default: :obj:`1`)
        epoch_select (string, optional): :obj:`"test_max"` or :obj:`"val_max"`.
            (default: :obj:`"test_max"`)
        device (int, or torch.device, optional): Device for computation. (default: :obj:`None`)
        **kwargs (optional): Training and evaluation configs in :meth:`setup_train_config`.
        
    Examples
    --------
    >>> encoder = Encoder(...)
    >>> model = Contrastive(...)
    >>> evaluator = GraphUnsupervised(dataset, log_interval=10, device=0, p_lr = 0.001)
    >>> evaluator.evaluate(model, encoder)
    """
    
    def __init__(self, args, log_interval=1): #, **kwargs):  #reduction='sum'
        
        if args.dataset == 'esol':
            self.dataset = MoleculeNet(root='dataset/', name='esol')  # regression
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
            
        elif args.encoder=='spherenet':
            self.int_emb_size = args.int_emb_size
            self.basis_emb_size_dist = args.basis_emb_size_dist
            self.basis_emb_size_angle = args.basis_emb_size_angle
            self.basis_emb_size_torsion = args.basis_emb_size_torsion
            self.out_emb_channels = args.out_emb_channels
            self.num_spherical = args.num_spherical
            self.num_radial = args.num_radial
            self.envelope_exponent = args.envelope_exponent
            self.num_before_skip = args.num_before_skip
            self.num_after_skip = args.num_after_skip
            self.num_output_layers = args.num_output_layers
            self.use_node_features = args.use_node_features
                
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
        #print(self.target)
                
        
    def evaluate(self, fold_seed=None):
        setup_seed(self.seed)
        pred_head = self.proj
        if self.finetune:
            encoder = Encoder(self.args)
            encoder.load_state_dict(torch.load(self.model_path), strict=False)
            model = PredictionModel(encoder, pred_head, self.z_dim, self.out_dim, self.dropout_rate)
            
        else: 
            if self.encoder == 'schnet':
                model = SchNet(cutoff=self.cutoff, num_layers=self.num_layers, hidden_channels=self.z_dim, out_channels=self.out_dim,
                               num_filters=self.num_filters, num_gaussians=self.num_gaussians, dropout_rate=self.dropout_rate)
            elif self.encoder == 'spherenet':
                model = SphereNet(cutoff=self.cutoff, num_layers=self.num_layers, hidden_channels=self.z_dim, out_channels=self.out_dim, 
                                  int_emb_size=self.int_emb_size, basis_emb_size_dist=self.basis_emb_size_dist, 
                                  basis_emb_size_angle=self.basis_emb_size_angle, basis_emb_size_torsion=self.basis_emb_size_torsion,
                                  out_emb_channels=self.out_emb_channels, num_spherical=self.num_spherical, num_radial=self.num_radial, 
                                  envelope_exponent=self.envelope_exponent, num_before_skip=self.num_before_skip, num_after_skip=self.num_after_skip, 
                                  num_output_layers=self.num_output_layers, dropout_rate=self.dropout_rate, use_node_features=self.use_node_features)
        num_params = sum(p.numel() for p in model.parameters())
        print(f'#Params: {num_params}')
        if self.task_type == 'cls':
            train_scores, train_losses, val_scores, val_losses, test_scores, test_losses = [], [], [], [], [], [] # Fold 단위
            for fold, train_loader, val_loader, test_loader in k_scaffold(self.n_folds, self.dataset, self.batch_size, self.task_type):
                fold_model = copy.deepcopy(model).to(self.device)
                if self.f_optim == 'StepLR':
                    f_optimizer = Adam(fold_model.parameters(), lr=self.f_lr, weight_decay=self.f_weight_decay)
                    scheduler = StepLR(f_optimizer, step_size=self.f_lr_decay_step_size, gamma=self.f_lr_decay_factor)
                if self.f_optim == 'ExponentialLR':
                    f_optimizer = Adam(fold_model.parameters(), lr=self.f_lr, weight_decay=self.f_weight_decay)
                    scheduler = ExponentialLR(f_optimizer, gamma=self.expo_gamma)    
                elif self.f_optim == 'Cosine':
                    f_optimizer = Adam(fold_model.parameters(), lr=0)
                    scheduler = CosineAnnealingWarmUpRestarts(f_optimizer, T_0=10, T_mult=1, eta_max=0.01,  T_up=10, gamma=0.5)
                wait = 0
                best_val_auc = 0  
                best_test_auc = 0 
                best_val_loss = float("inf") 
                best_test_loss = float("inf") 
                patience = 50     
                with trange(self.f_epoch) as t:
                    for epoch in t:
                        t.set_description('Finetune: epoch %d' % (epoch+1))

                        train_cls(fold_model, self.device, train_loader, f_optimizer)
                        train_auc, train_loss = eval_cls(fold_model, self.device, train_loader)
                        val_auc, val_loss = eval_cls(fold_model, self.device, val_loader)
                        test_auc, test_loss = eval_cls(fold_model, self.device, test_loader)
                        scheduler.step()
                        
                        # Adapted from https://github.com/PetarV-/GAT/blob/master/execute_cora.py
                        # https://github.com/graphdml-uiuc-jlu/geom-gcn/blob/master/train_GAT.py
                        if np.less(val_loss, best_val_loss):
                            best_val_auc = val_auc
                            best_test_auc = test_auc
                            wait = 0
                        
                        else:
                            wait += 1
                            if wait >= patience:
                                print('Early stop at Epoch: {:d} with final val auc: {:.4f}'.format(epoch, val_auc))
                                break
                        t.set_postfix(atrain_auc='{:.3f}'.format(float(train_auc)), 
                                      best_val_auc='{:.3f}'.format(float(best_val_auc)), 
                                      best_test_auc='{:.3f}'.format(float(best_test_auc)),
                                      ctrain_loss='{:.3f}'.format(float(train_loss)), 
                                      eval_loss='{:.3f}'.format(float(val_loss)), 
                                      dtest_loss='{:.3f}'.format(float(test_loss)),
                                      FOLD='{:.1f}'.format(fold))
                    val_scores.append(best_val_auc)
                    test_scores.append(best_test_auc)
                    if best_test_auc < 0.5:
                        break
            auc = np.mean(test_scores)
            sd = np.std(test_scores)
            return auc, sd
                
        elif self.task_type == 'reg':
            writer = SummaryWriter()
            # log_dir='./results/' + self.args.dataset + '/' + self.encoder
            train_losses, val_losses, test_losses = [], [], [] # Fold 단위
            best_val_losses, best_test_losses = [], []
            for i in range(self.n_times):
                for fold, train_loader, val_loader, test_loader in k_scaffold(self.n_folds, self.dataset, self.batch_size, self.task_type):
                    fold_model = copy.deepcopy(model).to(self.device)
                    if self.f_optim == 'StepLR':
                        f_optimizer = Adam(fold_model.parameters(), lr=self.f_lr, weight_decay=self.f_weight_decay)
                        scheduler = StepLR(f_optimizer, step_size=self.f_lr_decay_step_size, gamma=self.f_lr_decay_factor)
                    elif self.f_optim == 'ExponentialLR':
                        f_optimizer = Adam(fold_model.parameters(), lr=self.f_lr, weight_decay=self.f_weight_decay)
                        scheduler = ExponentialLR(f_optimizer, gamma=self.expo_gamma)    
                    elif self.f_optim == 'Cosine':
                        f_optimizer = Adam(fold_model.parameters(), lr=0)
                        scheduler = CosineAnnealingWarmUpRestarts(f_optimizer, T_0=10, T_mult=1, eta_max=0.01,  T_up=10, gamma=0.5)

                    wait = 0
                    best_val_loss = float("inf") 
                    best_test_loss = float("inf") 
                    patience = 999
                    fold_train_losses, fold_val_losses, fold_test_losses = [], [], [] # Fold 단위
                    with trange(self.f_epoch) as t:
                        for epoch in t:
                            t.set_description('Finetune: epoch %d' % (epoch+1))
                            train_reg(fold_model, self.device, train_loader, f_optimizer, mae=self.mae, target=self.target)
                            train_rmse = eval_reg(fold_model, self.device, train_loader, mae=self.mae, target=self.target)
                            val_rmse = eval_reg(fold_model, self.device, val_loader, mae=self.mae, target=self.target)
                            test_rmse = eval_reg(fold_model, self.device, test_loader, mae=self.mae, target=self.target)
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
                        plt.style.use('seaborn')
                        plt.plot(fold_train_losses, label = 'Training error')
                        plt.plot(fold_val_losses, label = 'Validation error')
                        plt.plot(fold_test_losses, label = 'Test error')
                        plt.ylabel('Loss', fontsize = 14)
                        plt.xlabel('Epoch', fontsize = 14)
                        plt.title('Learning curves for a fine-tuning model', fontsize = 18) # , y = 1.03
                        plt.legend()
                        plt.show()
                        best_val_losses.append(best_val_loss)
                        best_test_losses.append(best_test_loss)
                        

            directory = './results/' + self.args.dataset + '/' + self.encoder
            txtfile = directory + '/' + 'losses.txt'
            with open(txtfile, "a") as myfile:
                if self.finetune:
                    model = self.args.model_path.split('/')[-1]
                    parameter = self.args.model_path.split('/')[-2]
                    myfile.write('model:'+ str(model) + '_' + str(parameter) + '_')
                myfile.write('finetune:'+ str(self.finetune) + '_folds:' + str(self.n_folds) + '_target:' + str(self.target) + '_batch:' + str(self.batch_size) + '_cutoff:' + str(self.cutoff) + '_layer:' + str(self.num_layers) + '_filter:' + str(self.num_filters) + '_gau:' + str(self.num_gaussians) + '_z_dim:' + str(self.z_dim) + '_dropout:' + str(self.dropout_rate) + '_lr:' + str(self.f_lr) + '_weight_decay:' + str(self.f_weight_decay) + '_expo_gamma:' +  str(self.expo_gamma) + "\n") 
                myfile.write('train_losses:'+ str(train_losses) + '_val_losses:' + str(val_losses) + '_test_losses:' + str(test_losses) + '_best_val_losses:'+ str(best_val_losses) + '_best_test_losses:' + str(best_test_losses) + "\n" + "\n") 
            loss = np.mean(best_test_losses)
            sd = np.std(best_test_losses)
            #print(train_losses)
            #train_losses = torch.tensor(train_losses).view(self.n_times, self.n_folds)
            #val_losses = torch.tensor(val_losses).view(self.n_times, self.n_folds)
            #test_losses = torch.tensor(test_losses).view(self.n_times, self.n_folds)
            #print(train_losses)
            #print(val_losses)
            #print(test_losses)
            #return train_losses, val_losses, test_losses, best_val_losses, best_test_losses
            return loss, sd
        
        elif self.task_type == 'qm9':
            split_idx = self.dataset.get_idx_split(len(self.dataset.data.y), train_size=110000, valid_size=1000, seed=42)
            train_dataset, valid_dataset, test_dataset = self.dataset[split_idx['train']], self.dataset[split_idx['valid']],\
                                                         self.dataset[split_idx['test']]
            train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True)
            val_loader = DataLoader(valid_dataset, self.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, self.batch_size, shuffle=False)
            
            fold_model = copy.deepcopy(model).to(self.device)
            if self.f_optim == 'StepLR':
                f_optimizer = Adam(fold_model.parameters(), lr=self.f_lr, weight_decay=self.f_weight_decay)
                scheduler = StepLR(f_optimizer, step_size=self.f_lr_decay_step_size, gamma=self.f_lr_decay_factor)
            elif self.f_optim == 'ExponentialLR':
                f_optimizer = Adam(fold_model.parameters(), lr=self.f_lr, weight_decay=self.f_weight_decay)
                scheduler = ExponentialLR(f_optimizer, gamma=self.expo_gamma)    
            elif self.f_optim == 'Cosine':
                f_optimizer = Adam(fold_model.parameters(), lr=0)
                scheduler = CosineAnnealingWarmUpRestarts(f_optimizer, T_0=10, T_mult=1, eta_max=0.01,  T_up=10, gamma=0.5)

            wait = 0
            best_val_loss = float("inf") 
            best_test_loss = float("inf") 
            patience = 50
            with trange(self.f_epoch) as t:
                for epoch in t:
                    t.set_description('Finetune: epoch %d' % (epoch+1))
                    train_reg(fold_model, self.device, train_loader, f_optimizer, mae=True)
                    train_mae = eval_reg(fold_model, self.device, train_loader, mae=True)
                    val_mae = eval_reg(fold_model, self.device, val_loader, mae=True)
                    test_mae = eval_reg(fold_model, self.device, test_loader, mae=True)
                    scheduler.step()

                    if np.less(val_mae, best_val_loss):
                        best_val_loss = val_mae
                        best_test_loss = test_mae
                        wait = 0
                    else:
                        wait += 1
                        if wait >= patience:
                            print('Early stop at Epoch: {:d} with final val loss: {:.4f}'.format(epoch+1, val_mae))
                            break
                    t.set_postfix(best_val_loss='{:.3f}'.format(float(best_val_loss)), 
                                  best_test_loss='{:.3f}'.format(float(best_test_loss)),
                                  train_mae='{:.3f}'.format(float(train_mae)), 
                                  val_mae='{:.3f}'.format(float(val_mae)), 
                                  test_mae='{:.3f}'.format(float(test_mae)))
                val_losses.append(best_val_loss)
                test_losses.append(best_test_loss)
        loss = np.mean(test_losses)
        sd = np.std(test_losses)
        return loss, sd

    
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
        
        if self.task_type == 'cls':
            auc_m_lst = []
            auc_sd_lst = []
            paras = []
            for batch_size in batch_lst:
                for cutoff in cutoff_lst:
                    for num_layers in num_layers_lst:
                        for z_dim in z_dim_lst:
                            for dropout_rate in dropout_rate_lst:
                                self.setup_train_config(batch_size=batch_size, cutoff=cutoff, num_layers=num_layers, 
                                                        z_dim=z_dim, dropout_rate=dropout_rate)
                                auc_m, auc_sd = self.evaluate()
                                print('auc_m', auc_m)
                                auc_m_lst.append(auc_m)
                                auc_sd_lst.append(auc_sd)
                                paras.append((batch_size, cutoff, num_layers, z_dim, dropout_rate, target))
                                if self.f_optim == 'StepLR':
                                    with open(txtfile, "a") as myfile:
                                        if finetune:
                                            myfile.write('model:'+ str(model) + '_' + str(parameter))
                                        myfile.write('finetune:'+ str(finetune) + '_folds:' + str(n_folds) + '_target:' + str(target) + '_auc:'+ str(np.round(auc_m, 3)) + '_batch:' + str(batch_size) + '_cutoff:' + str(cutoff) + '_layer:' + str(num_layers) + '_filter:' + str(self.num_filters) + '_gau:' + str(self.num_gaussians) + '_z_dim:' + str(z_dim) + '_dropout:' + str(dropout_rate) + '_lr:' + str(self.f_lr) + '_weight_decay:' + str(self.f_weight_decay) + '_lr_decay_step_size:' +  str(self.f_lr_decay_step_size) + '_lr_decay_factor:' + str(self.f_lr_decay_factor) + "\n")
                                elif self.f_optim == 'ExponentialLR':
                                    with open(txtfile, "a") as myfile:
                                        if finetune:
                                            myfile.write('model:'+ str(model) + '_' + str(parameter))
                                        myfile.write('finetune:'+ str(finetune) + '_folds:' + str(n_folds) + '_target:' + str(target) + '_auc:'+ str(np.round(auc_m, 3)) + '_batch:' + str(batch_size) + '_cutoff:' + str(cutoff) + '_layer:' + str(num_layers) + '_filter:' + str(self.num_filters) + '_gau:' + str(self.num_gaussians) + '_z_dim:' + str(z_dim) + '_dropout:' + str(droposut_rate) + '_lr:' + str(self.f_lr) + '_weight_decay:' + str(self.f_weight_decay) + '_expo_gamma:' +  str(self.expo_gamma) + "\n")           
                                elif self.f_optim == 'Cosine':
                                    with open(txtfile, "a") as myfile:
                                        if finetune:
                                            myfile.write('model:'+ str(model) + '_' + str(parameter))
                                        myfile.write('finetune:'+ str(finetune) + '_folds:' + str(n_folds) + '_target:' + str(target) + '_auc:'+ str(np.round(auc_m, 3)) + '_batch:' + str(batch_size) + '_cutoff:' + str(cutoff) + '_layer:' + str(num_layers) + '_filter:' + str(self.num_filters) + '_gau:' + str(self.num_gaussians) + '_z_dim:' + str(z_dim) + '_dropout:' + str(dropout_rate) + '_T_0:' + str(self.T_0) + '_T_mult:' + str(self.T_mult) + '_eta_max:' +  str(self.eta_max) + '_T_up:' + str(self.T_up) +'_gamma:' + str(self.gamma) + "\n")
                                    
            idx = np.argmax(auc_m_lst)
            total = [(param, auc) for param, auc in zip(paras, auc_m_lst)]
            print('Total Results: ', total)
            return auc_m_lst[idx], auc_sd_lst[idx], paras[idx], total
        
        elif self.task_type=='reg':
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
                                                    self.setup_train_config(batch_size=batch_size, cutoff=cutoff, num_layers=num_layers, 
                                                                            num_filters=num_filters, num_gaussians=num_gaussians, z_dim=z_dim, 
                                                                            dropout_rate=dropout_rate, target=target, f_lr=f_lr, 
                                                                            f_weight_decay=f_weight_decay)
                                                    loss_m, loss_sd = self.evaluate()
                                                    loss_m_lst.append(loss_m)
                                                    loss_sd_lst.append(loss_sd)
                                                    paras.append((batch_size, cutoff, num_layers, z_dim, dropout_rate))
                                        
            idx = np.argmin(loss_m_lst)
            total = [(param, loss) for param, loss in zip(paras, loss_m_lst)]
            print('Total Results: ', total)
            return loss_m_lst[idx], loss_sd_lst[idx], paras[idx], total

        elif self.task_type=='qm9':
            loss_m_lst = []
            loss_sd_lst = []
            paras = []
            for batch_size in batch_lst:
                for cutoff in cutoff_lst:
                    for num_layers in num_layers_lst:
                        for z_dim in z_dim_lst:
                            for dropout_rate in dropout_rate_lst:
                                self.setup_train_config(batch_size=batch_size, cutoff=cutoff, 
                                                        num_layers=num_layers, z_dim=z_dim, dropout_rate=dropout_rate)
                                loss_m, loss_sd = self.evaluate()
                                loss_m_lst.append(loss_m)
                                loss_sd_lst.append(loss_sd)
                                paras.append((batch_size, cutoff, num_layers, z_dim, dropout_rate))
                                if self.f_optim == 'StepLR':

                                        
            idx = np.argmin(loss_m_lst)
            total = [(param, loss) for param, loss in zip(paras, loss_m_lst)]
            print('Total Results: ', total)

            return loss_m_lst[idx], loss_sd_lst[idx], paras[idx], total

    
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
    
    def __init__(self, encoder, pred_head, dim, out_dim, dropout):
        
        super(PredictionModel, self).__init__()
        self.encoder = encoder
        
        if pred_head == 'schnet':
            self.pred_head = nn.Sequential(nn.Linear(dim, dim//2),
                                 #nn.ReLU(inplace=True),
                                 ShiftedSoftplus(),
                                 nn.Dropout(dropout),
                                 nn.Linear(dim//2, out_dim))
            #self.sigmoid = nn.Sigmoid()
            
        elif pred_head == 'spherenet':
            self.pred_head = nn.Sequential(nn.Linear(dim, dim//2),
                                 nn.ReLU(inplace=True),
                                 #swish(),
                                 nn.Dropout(dropout),
                                 nn.Linear(dim//2, out_dim))
            #self.pred_head = nn.Linear(dim, out_dim)
        
    def forward(self, data):
        
        zg = self.encoder(data)
        out = self.pred_head(zg)
        return out
                      
                      
def checkbinary(yslist):
    check = []
    for ys in yslist:
        if 1 & -1 in ys:  # pass
            check.append(1)
        else:            # fold 한번 더
            check.append(2)
    return check    


def k_scaffold(n_folds, dataset, batch_size, task_type):
    #setup_seed(seed)
    i = 0
    #print(dataset.data)
    while i < n_folds:
        train, val, test = scaffold_split(dataset)
        
        if len(val) == 0 or len(test) == 0:
            continue
        train_loader = DataLoader(train, batch_size, shuffle=True)
        val_loader = DataLoader(val, batch_size, shuffle=False)
        test_loader = DataLoader(test, batch_size, shuffle=False)
        if len(val_loader) != len(test_loader):
            continue
        train_ys = [ data.y for idx, data in enumerate(train_loader) ]
        val_ys = [ data.y for idx, data in enumerate(val_loader) ]
        test_ys = [ data.y for idx, data in enumerate(test_loader) ]
        if task_type == 'cls':
            checklist = checkbinary(train_ys) + checkbinary(val_ys) + checkbinary(test_ys)
            #print(checklist)
            if 2 in checklist :
                continue
            else:
            #if 2 not in checklist or len(val)!=0 or len(test)!=0:  
                i+=1
            yield i, train_loader, val_loader, test_loader
        elif task_type == 'reg':
            print(len(train), len(val), len(test))
            if len(val)!=0 or len(test)!=0:
                i+=1
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
