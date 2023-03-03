import os
import torch
from tqdm import trange
import torch.nn as nn

from torch_geometric.nn.acts import swish
from torch_geometric.data import Batch, Data

from dig.sslgraph.method.contrastive.objectives import NCE_loss, JSE_loss
from dig.sslgraph.utils.encoders import ShiftedSoftplus
import matplotlib.pyplot as plt


class Contrastive(nn.Module):
    
    def __init__(self, args, objective, views_fn,
                 graph_level=True,
                 z_dim=None,
                 proj=None,
                 neg_by_crpt=False,
                 tau=None,
                 dropout=None,
                 device=None,
                 choice_model='best'):

        assert graph_level is not None
        assert not (objective=='NCE' and neg_by_crpt)

        super(Contrastive, self).__init__()
        self.args = args
        
        self.loss_fn = self._get_loss(objective)
        self.views_fn = views_fn # fn: (batched) graph -> graph
        self.graph_level = graph_level
        self.z_dim = z_dim
        self.proj = proj
        self.neg_by_crpt = neg_by_crpt
        self.tau = args.tau
        self.pc = args.pc
        self.dropout = dropout
        self.choice_model = choice_model
        self.model_path = args.model_path
        self.device = args.device
        self.encoder = args.encoder
        self.num_layers = args.num_layers
        
    def train(self, encoder, data_loader, optimizer, scheduler, epochs, per_epoch_out=False):
        r"""Perform contrastive training and yield trained encoders per epoch or after
        the last epoch.
        
        Args:
            encoder (Module, or list of Module): A graph encoder shared by all views or a list 
                of graph encoders dedicated for each view. If :obj:`node_level` = :obj:`False`, 
                the encoder should return tensor of shape [:obj:`n_graphs`, :obj:`z_dim`].
                Otherwise, return tuple of shape ([:obj:`n_graphs`, :obj:`z_dim`], 
                [:obj:`n_nodes`, :obj:`z_n_dim`]) representing graph-level and node-level embeddings.
            dataloader (Dataloader): Dataloader for unsupervised learning or pretraining.
            optimizer (Optimizer): Pytorch optimizer for trainable parameters in encoder(s).
            epochs (int): Number of total training epochs.
            per_epoch_out (bool): If True, yield trained encoders per epoch. Otherwise, only yield
                the final encoder at the last epoch. (default: :obj:`False`)
                
        :rtype: :class:`generator`.
        """
        self.per_epoch_out = per_epoch_out
        if self.graph_level and self.proj is not None:
            self.proj_head_g = self._get_proj(self.encoder, self.num_layers, self.proj, self.z_dim).to(self.device)
            optimizer.add_param_group({"params": self.proj_head_g.parameters()})
        if isinstance(encoder, list):
            encoder = [enc.to(self.device) for enc in encoder]
        else:
            encoder = encoder.to(self.device)
        train_fn = self.train_encoder_graph
            
        for enc in train_fn(self.args, encoder, data_loader, optimizer, scheduler, epochs):
            yield enc


    def train_encoder_graph(self, args, encoder, data_loader, optimizer, scheduler, epochs):
        if isinstance(encoder, list):
            assert len(encoder)==len(self.views_fn)
            encoders = encoder
            [enc.train() for enc in encoders]
        else:
            encoder.train()
            encoders = [encoder]*len(self.views_fn)

        try:
            self.proj_head_g.train()
        except:
            pass
        
        min_loss = 1e9
        losses = []
        with trange(epochs) as t:
            #current_path = os.getcwd()
            if args.p_optim == 'StepLR':
                directory = self.model_path + "/" +\
                                'enc_pretrain-%s_batch-%s_encoder-%s_proj-%s_cutoff-%s_layers-%s_filter-%s_gau-%s_z_dim-%s_lr-%s_aug_1-%s_aug_2-%s_aug_ratio-%s_tau-%s_optim-%s_weight_decay-%s_lr_decay_step_size-%s_lr_decay_factor-%s_dropout-%s'\
                               %(args.pretrain_dataset, args.batch_size, args.encoder, args.proj, args.cutoff, args.num_layers,
                                 args.num_filters, args.num_gaussians, args.z_dim, args.p_lr, args.aug_1, args.aug_2, args.aug_ratio, args.tau, 
                                 args.p_optim, args.p_weight_decay, args.p_lr_decay_step_size, args.p_lr_decay_factor, args.dropout_rate)
            elif args.p_optim == 'ExponentialLR':
                directory = self.model_path + "/" +\
                                'encoder-%s_pretrain-%s_batch-%s_proj-%s_cutoff-%s_layers-%s_filter-%s_gau-%s_z_dim-%s_lr-%s_aug_1-%s_aug_2-%s_aug_ratio-%s_tau-%s_optim-%s_weight_decay-%s_expo_gamma-%s_dropout-%s'\
                               %(args.encoder, args.pretrain_dataset, args.batch_size, args.proj, args.cutoff, args.num_layers, 
                                 args.num_filters, args.num_gaussians, args.z_dim, args.p_lr, args.aug_1, args.aug_2, args.aug_ratio, args.tau, 
                                 args.p_optim, args.p_weight_decay, args.expo_gamma, args.dropout_rate)
            elif args.p_optim == 'Cosine':
                directory = self.model_path + "/" +\
                                'enc_pretrain-%s_batch-%s_encoder-%s_proj-%s_node_features-%s_cutoff-%s_layers-%s_filter-%s_gau-%s_z_dim-%s_aug_1-%s_aug_2-%s_aug_ratio-%s_tau-%s_optim-%s_dropout-%s_T_0-%s_T_mult-%s_eta_max-%s_T_up-%s_gamma-%s'\
                               %(args.pretrain_dataset, args.batch_size, args.encoder, args.proj, args.use_node_features, args.cutoff, args.num_layers, 
                                 args.num_filters, args.num_gaussians, args.z_dim, args.p_lr, args.aug_1, args.aug_2, args.aug_ratio, args.tau, 
                                 args.p_optim, args.dropout_rate, args.T_0, args.T_mult, args.eta_max, args.T_up, args.gamma)
            print(directory)
            try:
                if not os.path.exists(directory):
                    os.makedirs(directory)
            except OSError:
                print ('Error: Creating directory. ' +  directory)
            txtfile = directory +'/' + 'loss.txt'

            for epoch in t:
                epoch_loss = 0.0
                t.set_description('Pretraining: epoch %d' % (epoch+1))
                for data in data_loader:
                    optimizer.zero_grad()
                    data = data.to(self.device)
                    if None in self.views_fn: 
                        # For view fn that returns multiple views
                        views = []
                        for v_fn in self.views_fn:
                            if v_fn is not None:
                                views += [*v_fn(data)]
                    else:
                        views = [v_fn(data) for v_fn in self.views_fn]
                    

                    zs = []
                    for view, enc in zip(views, encoders):
                        z = self._get_embed(enc, view.to(self.device))
                        zs.append(self.proj_head_g(z))
                    loss = self.loss_fn(zs, neg_by_crpt=self.neg_by_crpt, tau=self.tau, 
                                        pc=self.pc)
                                        
                                        #E0=views[0].min_energy, E1=views[0].energy, E2=views[1].energy)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss
                    scheduler.step()
                losses.append(epoch_loss)
                
                t.set_postfix(loss='{:.4f}'.format(float(epoch_loss)))
                #torch.save(encoder.state_dict(), directory + '/enc_epoch-%d_loss-%.3f.pkl'%(epoch+1, epoch_loss))
                
                if self.choice_model == 'best' and epoch_loss < min_loss:
                    min_loss = epoch_loss
                    
                    if not os.path.exists(self.model_path):
                        try:
                            os.mkdir(self.model_path)
                        except:
                            raise RuntimeError('cannot create model path')

                    if isinstance(encoder, list):
                        for i, enc in enumerate(encoder):
                            torch.save(enc.state_dict(), self.model_path+'/enc%d_best.pkl'%i)
                    else:
                        torch.save(encoder.state_dict(), directory +'/enc_best_epoch-%d_loss-%.3f.pkl'%(epoch+1, epoch_loss))
                        
                #if self.per_epoch_out:
                 #   yield encoder, self.proj_head_g
            
        plt.style.use('seaborn')
        plt.plot(losses, label = 'Pretrain error')
        plt.ylabel('Loss', fontsize = 14)
        plt.xlabel('Epoch', fontsize = 14)
        plt.title('Learning curves for a pre-training model', fontsize = 18) # , y = 1.03
        plt.legend()
        plt.show()
        if not self.per_epoch_out:
            print('last')
            yield encoder, self.proj_head_g

    
    
    def _get_embed(self, enc, view):
        
        if self.neg_by_crpt:
            view_crpt = self._corrupt_graph(view)
            if self.node_level and self.graph_level:
                z_g, z_n = enc(view)
                z_g_crpt, z_n_crpt = enc(view_crpt)
                z = (torch.cat([z_g, z_g_crpt], 0),
                     torch.cat([z_n, z_n_crpt], 0))
            else:
                z = enc(view)
                z_crpt = enc(view_crpt)
                z = torch.cat([z, z_crpt], 0)
        else:
            z = enc(view)
        
        return z
                     
    
    def _get_proj(self, encoder, num_layers, proj_head, z_dim):
        
        if callable(proj_head):
            return proj_head
        
        assert proj_head in ['schnet', 'spherenet']
        
       
        if proj_head == 'schnet':
            
            proj_nn = nn.Sequential(nn.Linear(z_dim, z_dim),
                                 #nn.ReLU(inplace=True),
                                 ShiftedSoftplus(),
                                 nn.Dropout(self.dropout),
                                 nn.Linear(z_dim, z_dim))
            for m in proj_nn.modules():
                self._weights_init(m)

        elif proj_head == 'spherenet':
            init_z_dim = num_layers*z_dim if 'gin' in encoder or 'gcn' in encoder else z_dim
            proj_nn = nn.Sequential(nn.Linear(init_z_dim, z_dim),
                                 nn.ReLU(inplace=True),
                                 #swish(),
                                 nn.Dropout(self.dropout),
                                 nn.Linear(z_dim, z_dim))
            for m in proj_nn.modules():
                self._weights_init(m)
                
                
        elif proj_head == 'linear':
            proj_nn = nn.Linear(in_dim, out_dim)
            self._weights_init(proj_nn)
            
        return proj_nn
        
    def _weights_init(self, m):        
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        
    def _get_loss(self, objective):
        
        if callable(objective):
            return objective
        
        assert objective in ['JSE', 'NCE']
        
        return {'JSE':JSE_loss, 'NCE':NCE_loss}[objective]
