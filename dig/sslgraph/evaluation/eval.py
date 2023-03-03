import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import trange, tqdm
from sklearn.model_selection import StratifiedKFold
from dig.threedgraph.dataset.dataset import scaffold_split
from dig.sslgraph.utils.seed import setup_seed

from torch_geometric.loader import DataLoader

from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch
from torch.utils import data as torch_data



criterion = nn.BCEWithLogitsLoss(reduction = "none")

def train_cls(model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()
        optimizer.step()
        #print('trainloss', loss.item())


def train_reg(model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        loss = torch.sum((pred-y)**2)/y.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval_cls(model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
    roc_list = []
    #loss_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))
        #loss_list.append(log_loss(y_true.cpu(), y_scores.cpu(), eps=1e-6))
    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list)


def eval_reg(model, device, loader):
    model.eval()
    y_true = []
    y_scores = []
    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)
    
    y_true = torch.cat(y_true, dim = 0).cpu().numpy().flatten()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy().flatten()
    rmse = mean_squared_error(y_true, y_scores, squared=False)
    cor = pearsonr(y_true, y_scores)[0]
    print(rmse, cor)
    return rmse, cor


class GraphUnsupervised(object):
    r"""
    The evaluation interface for unsupervised graph representation learning evaluated with 
    linear classification. You can refer to `the benchmark code 
    <https://github.com/divelab/DIG/tree/dig/benchmarks/sslgraph>`_ 
    for examples of usage.
    
    Args:
        dataset (torch_geometric.data.Dataset): The graph classification dataset.
        classifier (string, optional): Linear classifier for evaluation, :obj:`"SVC"` or 
            :obj:`"LogReg"`. (default: :obj:`"SVC"`)
        log_interval (int, optional): Perform evaluation per k epochs. (default: :obj:`1`)
        epoch_select (string, optional): :obj:`"test_max"` or :obj:`"val_max"`.
            (default: :obj:`"test_max"`)
        n_folds (int, optional): Number of folds for evaluation. (default: :obj:`10`)
        device (int, or torch.device, optional): Device for computation. (default: :obj:`None`)
        **kwargs (optional): Training and evaluation configs in :meth:`setup_train_config`.
        
    Examples
    --------
    >>> encoder = Encoder(...)
    >>> model = Contrastive(...)
    >>> evaluator = GraphUnsupervised(dataset, log_interval=10, device=0, p_lr = 0.001)
    >>> evaluator.evaluate(model, encoder)
    """
    
    def __init__(self, dataset_pretrain, dataset, out_dim, classifier='SVC', log_interval=1, epoch_select='val', 
                 split='scaffold', n_folds=5, device=None, **kwargs):  #reduction='sum'
        
        self.dataset, self.dataset_pretrain = dataset, dataset_pretrain
        #self.metric = metric
        #if self.metric == 'bce':
         #   self.loss = nn.BCEWithLogitsLoss()
        #elif self.metric == 'rmse':
            #self.loss = RMSELoss()
        self.epoch_select = epoch_select
        self.classifier = classifier
        self.log_interval = log_interval
        self.split = split
        self.n_folds = n_folds
        self.out_dim = out_dim
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, int):
            self.device = torch.device('cuda:%d'%device)
        else:
            self.device = device

        # Use default config if not further specified
        self.setup_train_config(**kwargs)

    def setup_train_config(self, batch_size = 64, 
                           p_optim = 'Adam', p_lr = 0.01, p_weight_decay = 0, p_epoch = 20, svc_search = True,
                           f_optim = 'Adam', f_lr = 0.001, f_weight_decay = 0, f_epoch = 100):
        r"""Method to setup training config.
        
        Args:
            batch_size (int, optional): Batch size for pretraining and inference. 
                (default: :obj:`256`)
            p_optim (string, or torch.optim.Optimizer class): Optimizer for pretraining.
                (default: :obj:`"Adam"`)
            p_lr (float, optional): Pretraining learning rate. (default: :obj:`0.01`)
            p_weight_decay (float, optional): Pretraining weight decay rate. 
                (default: :obj:`0`)
            p_epoch (int, optional): Pretraining epochs number. (default: :obj:`20`)
            svc_search (string, optional): If :obj:`True`, search for hyper-parameter 
                :obj:`C` in SVC. (default: :obj:`True`)
        """
        
        self.batch_size = batch_size

        self.p_optim = p_optim
        self.p_lr = p_lr
        self.p_weight_decay = p_weight_decay
        self.p_epoch = p_epoch
        
        self.search = svc_search

        self.f_optim = f_optim
        self.f_lr = f_lr
        self.f_weight_decay = f_weight_decay
        self.f_epoch = f_epoch
        
    
    def evaluate(self, learning_model, encoder, finetune=True, task_type ='cls', pred_head='MLP', fold_seed=None):
        r"""Run evaluation with given learning model and encoder(s).
        
        Args:
            learning_model: An object of a contrastive model (sslgraph.method.Contrastive) 
                or a predictive model.
            encoder (torch.nn.Module): Trainable pytorch model or list of models.
            fold_seed (int, optional): Seed for fold split. (default: :obj:`None`)

        :rtype: (float, float)
        """
        if finetune:
            pretrain_loader = DataLoader(self.dataset_pretrain, self.batch_size, shuffle=True)
            p_optimizer = self.get_optim(self.p_optim)(encoder.parameters(), lr=self.p_lr,
                                                      weight_decay=self.p_weight_decay)
            if self.p_epoch > 0:
                encoder = next(learning_model.train(encoder, pretrain_loader, p_optimizer, self.p_epoch))
            model = PredictionModel(encoder, pred_head, learning_model.z_dim, self.out_dim).to(self.device)
            val = not (self.epoch_select == 'test_max' or self.epoch_select =='test_min')  # 뭐지??
            train_scores, train_losses, val_scores, val_losses, test_scores, test_losses = [], [], [], [], [], []
            if task_type == 'cls':
                for fold, train_loader, test_loader, val_loader in k_scaffold(self.n_folds, self.dataset, self.batch_size, task_type):
                    fold_model = copy.deepcopy(model)
                    f_optimizer = self.get_optim(self.f_optim)(fold_model.parameters(), lr=self.f_lr,
                                                               weight_decay=self.f_weight_decay)
                    # training based on task type
                        #with open(txtfile, "a") as myfile:
                         #   myfile.write('epoch: train_auc val_auc test_auc\n')
                    wait = 0
                    best_val_auc = 0
                    best_test_auc = 0
                    patience = 10
                    with trange(self.f_epoch) as t:
                        for epoch in t:
                            print("====epoch " + str(epoch))

                            train_cls(fold_model, self.device, train_loader, f_optimizer)

                            print("====Evaluation")

                            train_auc = eval_cls(fold_model, self.device, train_loader)
                            val_auc = eval_cls(fold_model, self.device, val_loader)
                            test_auc = eval_cls(fold_model, self.device, test_loader)

                            #with open(txtfile, "a") as myfile:
                             #   myfile.write(str(int(epoch)) + ': ' + str(train_auc) + ' ' + str(val_auc) + ' ' + str(test_auc) + "\n")

                            print("train AUC: %f val AUC: %f test AUC: %f" %(train_auc, val_auc, test_auc))

                            # Early stopping
                            if np.greater(val_auc, best_val_auc):  # change for train loss
                                best_val_auc = val_auc
                                best_test_auc = test_auc

                                wait = 0
                            else:
                                wait += 1
                                if wait >= patience:
                                    print('Early stop at Epoch: {:d} with final val auc: {:.4f}'.format(epoch, val_auc))
                                    break

                    val_scores.append(best_val_auc)
                    test_scores.append(best_test_auc)
                    print(test_scores)
                auc = np.mean(test_scores)
                sd = np.std(test_scores)
                return auc, sd
                
                
            elif task_type == 'reg':
                for fold, train_loader, test_loader, val_loader in k_scaffold(self.n_folds, self.dataset, self.batch_size, task_type):
                    fold_model = copy.deepcopy(model)
                    f_optimizer = self.get_optim(self.f_optim)(fold_model.parameters(), lr=self.f_lr,
                                                               weight_decay=self.f_weight_decay)
                    # training based on task type
                        #with open(txtfile, "a") as myfile:
                         #   myfile.write('epoch: train_auc val_auc test_auc\n')
                    wait = 0
                    best_val_loss = float("inf") 
                    best_test_loss = float("inf") 
                    patience = 15
                    with trange(self.f_epoch) as t:
                        for epoch in t:
                            print("====epoch " + str(epoch))

                            train_reg(fold_model, self.device, train_loader, f_optimizer)

                            print("====Evaluation")

                            train_rmse, train_cor = eval_reg(fold_model, self.device, train_loader)
                            val_rmse, val_cor = eval_reg(fold_model, self.device, val_loader)
                            test_rmse, test_cor = eval_reg(fold_model, self.device, test_loader)

                            print("train rmse: %f val rmse: %f test rmse: %f" %(train_rmse, val_rmse, test_rmse))
                            print("train cor: %f val cor: %f test cor: %f" %(train_cor, val_cor, test_cor))

                            # Early stopping
                            if np.less(val_rmse, best_val_loss):
                                best_val_loss = val_rmse
                                best_test_loss = test_rmse
                                wait = 0
                            else:
                                wait += 1
                                if wait >= patience:
                                    print('Early stop at Epoch: {:d} with final val lss: {:.4f}'.format(epoch, val_rmse))
                                    break
            
                    val_losses.append(best_val_auc)
                    test_losses.append(best_test_loss)
                loss = np.mean(test_losses)
                sd = np,std(test_losses)
                return loss, sd

    def grid_search(self, learning_model, encoder, task_type, fold_seed=12345,
                    p_lr_lst=[0.01,0.001, 0.00001], p_epoch_lst=[5, 10, 30], f_lr_lst=[0.01,0.001, 0.00001], f_epoch_lst=[5, 10, 30, 50]):
        r"""Perform grid search on learning rate and epochs in pretraining.
        
        Args:
            learning_model: An object of a contrastive model (sslgraph.method.Contrastive) 
                or a predictive model.
            encoder (torch.nn.Module): Trainable pytorch model or list of models.
            p_lr_lst (list, optional): List of learning rate candidates.
            p_epoch_lst (list, optional): List of epochs number candidates.

        :rtype: (float, float, (float, int))
        """
        if task_type == 'cls':
            auc_m_lst = []
            auc_sd_lst = []
            paras = []
            for p_lr in p_lr_lst:
                for p_epoch in p_epoch_lst:
                    for f_lr in f_lr_lst:
                        for f_epoch in f_epoch_lst:
                            self.setup_train_config(p_lr=p_lr, p_epoch=p_epoch, f_lr=f_lr, f_epoch=f_epoch)
                            model = copy.deepcopy(learning_model)
                            enc = copy.deepcopy(encoder)
                            auc_m, auc_sd = self.evaluate(model, enc, fold_seed)
                            print('auc_m', auc_m)
                            auc_m_lst.append(auc_m)
                            auc_sd_lst.append(auc_sd)
                            paras.append((p_lr, p_epoch, f_lr, f_epoch))
            idx = np.argmax(auc_m_lst)
            print('Best paras: %d epoch, lr=%f, acc=%.4f' %(
                paras[idx][1], paras[idx][0], auc_m_lst[idx]))

            return auc_m_lst[idx], auc_sd_lst[idx], paras[idx]
        
        elif task_type=='reg':
            loss_m_lst = []
            loss_sd_lst = []
            paras = []
            for p_lr in p_lr_lst:
                for p_epoch in p_epoch_lst:
                    self.setup_train_config(p_lr=p_lr, p_epoch=p_epoch)
                    model = copy.deepcopy(learning_model)
                    enc = copy.deepcopy(encoder)
                    loss_m, loss_sd = self.evaluate(model, enc, fold_seed)
                    loss_m_lst.append(loss_m)
                    loss_sd_lst.append(loss_sd)
                    paras.append((p_lr, p_epoch))
            idx = np.argmin(loss_m_lst)
            print('Best paras: %d epoch, lr=%f, acc=%.4f' %(
                paras[idx][1], paras[idx][0], acc_m_lst[idx]))

            return loss_m_lst[idx], loss_sd_lst[idx], paras[idx]
            
    
    def get_embed(self, model, loader):
    
        model.eval()
        ret, y = [], []
        with torch.no_grad():
            for data in loader:
                y.append(data.y.numpy())
                data.to(self.device)
                embed = model(data)
                ret.append(embed.cpu().numpy())

        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y
        
        
    def get_clf(self):
        
        if self.classifier == 'SVC':
            return self.svc_clf
        elif self.classifier == 'LogReg':
            return self.log_reg
        else:
            return None
        
    
    def get_optim(self, optim):
        
        if callable(optim):
            return optim
        
        optims = {'Adam': torch.optim.Adam}
        
        return optims[optim]
    

    
class PredictionModel(nn.Module):
    
    def __init__(self, encoder, pred_head, dim, out_dim):
        
        super(PredictionModel, self).__init__()
        self.encoder = encoder
        
        if pred_head is not None:
            self.pred_head = nn.Sequential(nn.Linear(dim, dim//2),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(dim//2, out_dim))
            #for m in self.pred_head.modules():
            self.sigmoid = nn.Sigmoid()
             #   self._weights_init(m)
            
        else:
            self.pred_head = nn.Linear(dim, out_dim)
        
    def forward(self, data):
        
        zg = self.encoder(data)
        out = self.pred_head(zg)
        #out = out.view(-1)
        #print('before sigmoid out.shape', out.shape)
        #out = self.sigmoid(out)
        #print('out', out)
        #print('after sigmoid out.shape', out.shape)
        #print('torch.sigmoid(out)', torch.sigmoid(out))
        return out
        #return nn.functional.log_softmax(out, dim=-1)
                      
                      

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
    while i < n_folds:
        train, val, test = scaffold_split(dataset)
        if len(val) == 0 or len(test) == 0:
            continue
        train_loader = DataLoader(train, batch_size, shuffle=True)
        val_loader = DataLoader(val, batch_size, shuffle=False)
        test_loader = DataLoader(test, batch_size, shuffle=False)
        train_ys = [ data.y for idx, data in enumerate(train_loader) ]
        val_ys = [ data.y for idx, data in enumerate(val_loader) ]
        test_ys = [ data.y for idx, data in enumerate(test_loader) ]
        if task_type == 'cls':
            checklist = checkbinary(train_ys) + checkbinary(val_ys) + checkbinary(test_ys)
            if 2 in checklist :
                continue
            if 2 not in checklist or len(val)!=0 or len(test)!=0:  # val, test 길이가 0이 아니고 y에 0과 1이 같이 없으면  # and 2 not in checklist 
                i+=1
            yield i, train_loader, val_loader, test_loader
        elif task_type == 'reg':
            if len(val)!=0 or len(test)!=0:
                i+=1
            yield i, train_loader, val_loader, test_loader
   