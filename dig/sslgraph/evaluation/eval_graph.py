import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import trange
from sklearn.model_selection import StratifiedKFold
from dig.threedgraph.dataset.dataset import scaffold_split
from dig.sslgraph.utils.seed import setup_seed

from torch_geometric.loader import DataLoader

from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch
from torch.utils import data as torch_data


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + self.eps)

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


criterion = nn.BCEWithLogitsLoss(reduction = "none")

def train_cls(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
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


def train_reg(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        loss = torch.sum((pred-y)**2)/y.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval_cls(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

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

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]


def eval_reg(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy().flatten()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy().flatten()
    print(y_true.shape, y_scores.shape)
    mse = mean_squared_error(y_true, y_scores)
    cor = pearsonr(y_true, y_scores)[0]
    print(mse, cor)
    return mse, cor


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
    
    def __init__(self, dataset_pretrain, dataset, classifier='SVC', log_interval=1, epoch_select='val', 
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
        self.out_dim = 1
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
                #print('learning_model', learning_model)
                encoder = next(learning_model.train(encoder, pretrain_loader, p_optimizer, self.p_epoch))
                #print('encoder', encoder)
            model = PredictionModel(encoder, pred_head, learning_model.z_dim, self.out_dim).to(self.device)
            #print('model', model)
            val = not (self.epoch_select == 'test_max' or self.epoch_select =='test_min')  # 뭐지??
            print('val', val)
            train_scores, train_losses, val_scores, val_losses, test_scores, test_losses = [], [], [], [], [], []
            for fold, train_loader, test_loader, val_loader in k_scaffold(self.n_folds, self.dataset, self.batch_size):
                fold_model = copy.deepcopy(model)
                f_optimizer = self.get_optim(self.f_optim)(fold_model.parameters(), lr=self.f_lr,
                                                           weight_decay=self.f_weight_decay)
                if task_type ='cls':
                with trange(self.f_epoch) as t:
                    for epoch in t:
                        t.set_description('Fold %d, finetuning' % (fold))
                        train_score, train_loss = self.finetune(fold_model, f_optimizer, train_loader)
                        train_scores.append(train_score)
                        train_losses.append(train_loss)
                        val_score, val_loss = self.eval_val(fold_model, val_loader)
                        val_scores.append(val_score)
                        val_losses.append(val_loss)
                        test_score, test_loss = self.eval_val(fold_model, test_loader)
                        test_scores.append(test_score)
                        test_losses.append(test_loss)
                      
                        t.set_postfix(val_loss='{:.4f}'.format(val_loss),
                                      test_score='{:.4f}'.format(test_score))
                      
            
            train_scores, train_losses = torch.tensor(train_scores), torch.tensor(train_losses)
            val_scores, val_losses = torch.tensor(val_scores), torch.tensor(val_losses)
            test_scores, test_losses = torch.tensor(test_scores), torch.tensor(test_losses)
            train_scores, train_losses = train_scores.view(self.n_folds, self.f_epoch), train_losses.view(self.n_folds, self.f_epoch)
            val_scores, val_losses = val_scores.view(self.n_folds, self.f_epoch), val_losses.view(self.n_folds, self.f_epoch)
            test_scores, test_losses = test_scores.view(self.n_folds, self.f_epoch), test_losses.view(self.n_folds, self.f_epoch)
           
                      
            #if self.epoch_select == 'test_max':
            #    _, selection =  test_scores.mean(dim=0).max(dim=0)
            #    selection = selection.repeat(self.n_folds)
            #elif self.epoch_select == 'test_min':
            #    _, selection =  test_scores.mean(dim=0).min(dim=0)
            #    selection = selection.repeat(self.n_folds)
            #else:
            #    _, selection =  val_losses.min(dim=1)  # 가장 val loss가 낮은 epoch을 fold별로 추출
            # 가장 val loss가 낮은 epoch일 때의 score을 fold별로 추출
            #best_test_scores = test_scores[torch.arange(self.n_folds, dtype=torch.long), selection]  
            #test_auc_mean = best_test_scores.mean().item()
            #test_auc_std = best_test_scores.std().item() 
            #best_test_scores = test_scores[torch.arange(self.n_folds, dtype=torch.long), selection]  
            #test_auc_mean = best_test_scores.mean().item()
            #test_auc_std = best_test_scores.std().item()
            
            #return test_auc_mean, test_auc_std
            return train_scores, train_losses, val_scores, val_losses, test_scores, test_losses


        else:
            pretrain_loader = DataLoader(self.dataset_pretrain, self.batch_size, shuffle=True)      

            if isinstance(encoder, list):
                params = [{'params': enc.parameters()} for enc in encoder]
            else:
                params = encoder.parameters()

            p_optimizer = self.get_optim(self.p_optim)(params, lr=self.p_lr, 
                                                       weight_decay=self.p_weight_decay)

            test_scores_m, test_scores_sd = [], []
            for i, enc in enumerate(learning_model.train(encoder, pretrain_loader, 
                                                         p_optimizer, self.p_epoch, True)):
                #print(i)
                #print(encoder)
                ## 여기까지 Contrastive ##
                if (i+1)%self.log_interval==0:
                    ## 여기부터 지도학습 ##
                    test_scores = []
                    if self.split == 'scaffold':
                        loader = DataLoader(self.dataset, self.batch_size, shuffle=False)
                        embed, lbls = self.get_embed(enc.to(self.device), loader)
                        print(embed)
                        print(lbls)
                        lbs = np.array(preprocessing.LabelEncoder().fit_transform(lbls))
                        for i in range(3):  # random하게 3번 평가
                            train, val, test = scaffold_split(self.dataset)
                            train_index, test_index = train.indices, val.indices + test.indices  # 8:2
                            test_score = self.get_clf()(embed[train_index], lbls[train_index],
                                                        embed[test_index], lbls[test_index])
                            test_scores.append(test_score)
                    else:

                        #print(self.dataset)
                        loader = DataLoader(self.dataset, self.batch_size, shuffle=False)
                        #print(1)
                        embed, lbls = self.get_embed(enc.to(self.device), loader)
                        lbs = np.array(preprocessing.LabelEncoder().fit_transform(lbls))

                        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=fold_seed)
                        for fold, (train_index, test_index) in enumerate(kf.split(embed, lbls)):
                            test_score = self.get_clf()(embed[train_index], lbls[train_index],
                                                        embed[test_index], lbls[test_index])
                            test_scores.append(test_score)

                    kfold_scores = torch.tensor(test_scores)
                    test_score_mean = kfold_scores.mean().item()
                    test_score_std = kfold_scores.std().item() 
                    test_scores_m.append(test_score_mean)
                    test_scores_sd.append(test_score_std)
            idx = np.argmax(test_scores_m)
            acc = test_scores_m[idx]
            sd = test_scores_sd[idx]
            print('Best epoch %d: acc %.4f +/-(%.4f)'%((idx+1)*self.log_interval, acc, sd))
            return acc, sd 


    def grid_search(self, learning_model, encoder, fold_seed=12345,
                    p_lr_lst=[0.1,0.01,0.001], p_epoch_lst=[20,40,60]):
        r"""Perform grid search on learning rate and epochs in pretraining.
        
        Args:
            learning_model: An object of a contrastive model (sslgraph.method.Contrastive) 
                or a predictive model.
            encoder (torch.nn.Module): Trainable pytorch model or list of models.
            p_lr_lst (list, optional): List of learning rate candidates.
            p_epoch_lst (list, optional): List of epochs number candidates.

        :rtype: (float, float, (float, int))
        """
        
        acc_m_lst = []
        acc_sd_lst = []
        paras = []
        for p_lr in p_lr_lst:
            for p_epoch in p_epoch_lst:
                self.setup_train_config(p_lr=p_lr, p_epoch=p_epoch)
                model = copy.deepcopy(learning_model)
                enc = copy.deepcopy(encoder)
                acc_m, acc_sd = self.evaluate(model, enc, fold_seed)
                acc_m_lst.append(acc_m)
                acc_sd_lst.append(acc_sd)
                paras.append((p_lr, p_epoch))
        idx = np.argmax(acc_m_lst)
        print('Best paras: %d epoch, lr=%f, acc=%.4f' %(
            paras[idx][1], paras[idx][0], acc_m_lst[idx]))
        
        return acc_m_lst[idx], acc_sd_lst[idx], paras[idx]

                      
    def finetune(self, model, optimizer, loader):
        
        model.train()
        loss_accum = 0
        auc_accum = 0
        idx = 0
        for i, data in enumerate(loader):
            optimizer.zero_grad()
            data = data.to(self.device)
            out = model(data)
            loss = self.loss(out, data.y.view(-1, 1))
            auc = roc_auc_score(data.y.unsqueeze(1).cpu().detach().numpy(), out.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
            loss_accum += loss.detach().cpu().item()
            auc_accum += auc
            idx  = i
        return auc_accum / (idx + 1), loss_accum / (idx + 1)
                      
    def eval_val(self, model, loader, eval_mode=True):
        
        if eval_mode:
            model.eval()
                      
        loss = 0
        auc = 0
        idx = 0
        for i, data in enumerate(loader):
            data = data.to(self.device)
            with torch.no_grad():
                pred = model(data)
            loss += self.loss(pred, data.y.view(-1, 1)).item()
            auc += roc_auc_score(data.y.view(-1, 1).cpu(), pred.cpu())
            idx  = i
            
        
        return auc / int(idx+1), loss / int(idx+1)
        

    def eval_acc(self, model, loader, eval_mode=True):
        
        if eval_mode:
            model.eval()
                      
        correct = 0
        for data in loader:
            data = data.to(self.device)
            with torch.no_grad():
                pred = model(data).max(1)[1]
            correct += pred.eq(data.y.view(-1)).sum().item()
        return correct / len(loader.dataset)

                      
    def eval_auc(self, model, loader, eval_mode=True):
        
        if eval_mode:
            model.eval()
                      
        auc = 0
        idx = 0
        for i, data in enumerate(loader):
            data = data.to(self.device)
            with torch.no_grad():
                #pred = model(data).max(1)[1]
                pred = model(data)
                
                ones = torch.ones_like(data.y.view(-1))
            auc += roc_auc_score(data.y.view(-1, 1).cpu(), pred.cpu())
            idx  = i

        return auc / int(idx+1)
                      
                      
    def eval_metric(self, model, loader, eval_mode=True):

        if self.metric == 'acc':
            return self.eval_acc(model, loader, eval_mode)
        elif self.metric == 'bce':
            return self.eval_auc(model, loader, eval_mode)
        elif self.metric == 'rmse':
            return self.eval_loss(model, loader, eval_mode)
                      
    def svc_clf(self, train_embs, train_lbls, test_embs, test_lbls):

        if self.search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            #classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='roc_auc', verbose=0)
        else:
            classifier = SVC(C=10)

        classifier.fit(train_embs, train_lbls)
        acc = accuracy_score(test_lbls, classifier.predict(test_embs))
        auc = roc_auc_score(test_lbls, classifier.predict(test_embs))
        return auc
    
    
    def log_reg(self, train_embs, train_lbls, test_embs, test_lbls):
        
        train_embs = torch.from_numpy(train_embs).to(self.device)
        train_lbls = torch.from_numpy(train_lbls).to(self.device)
        test_embs = torch.from_numpy(test_embs).to(self.device)
        test_lbls = torch.from_numpy(test_lbls).to(self.device)

        xent = nn.CrossEntropyLoss()
        log = LogReg(hid_units, nb_classes)
        log.to(self.device)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        best_val = 0
        test_acc = None
        for it in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        
        return acc
    
    
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

        return out
        #return nn.functional.log_softmax(out, dim=-1)
                      
                      
#def k_scaffold(n_folds, dataset, batch_size):
    #setup_seed(seed)
    #for i in range(n_folds):
     #   if batch_size is None:
     #       batch_size = len()
     #   train, val, test = scaffold_split(dataset)
     #   train_loader = DataLoader(train, batch_size, shuffle=True)
     #   val_loader = DataLoader(val, batch_size, shuffle=False)
     #   test_loader = DataLoader(test, batch_size, shuffle=False)
                      
     #   yield i, train_loader, test_loader, val_loader

def checkbinary(yslist):
    check = []
    for ys in yslist:
        if 0 & 1 in ys:  # pass
            check.append(1)
        else:            # fold 한번 더
            check.append(2)
    return check    


def k_scaffold(n_folds, dataset, batch_size):
    #setup_seed(seed)
    i = 0
    while i < n_folds:
        train, val, test = scaffold_split(dataset)
        train_loader = DataLoader(train, batch_size, shuffle=True)
        val_loader = DataLoader(val, batch_size, shuffle=False)
        test_loader = DataLoader(test, batch_size, shuffle=False)
        train_ys = [ data.y for idx, data in enumerate(train_loader) ]
        val_ys = [ data.y for idx, data in enumerate(val_loader) ]
        test_ys = [ data.y for idx, data in enumerate(test_loader) ]
        checklist = checkbinary(train_ys) + checkbinary(val_ys) + checkbinary(test_ys)
        print(checklist)
        if 2 in checklist :
            continue
        print(i)
        if 2 not in checklist and len(val)!=0 and len(test)!=0:  # val, test 길이가 0이 아니고 y에 0과 1이 같이 없으면  # and 2 not in checklist 
            i+=1
        yield i, train_loader, val_loader, test_loader
   
        

def k_scaffold1(n_folds, dataset, batch_size):
    #setup_seed(seed)
    i = 0
    while i < n_folds:
        train, val, test = scaffold_split(dataset)
        valdata = val.dataset[val.indices]
        #print(valdata.data.y)
        #print(len(valdata.data.y))
        
        #print(val.indices)
        if len(val) !=0 and len(test) !=0:
            i+=1
            train_loader = DataLoader(train, batch_size, shuffle=True)
            val_loader = DataLoader(val, batch_size, shuffle=False)
            test_loader = DataLoader(test, batch_size, shuffle=False)
            idx = 0
            for idx, data in enumerate(val_loader):
                if 0 | 1 in data.y:
                    print(1)

                #if idx == 1:
                    #break

        yield i, train_loader, val_loader, test_loader