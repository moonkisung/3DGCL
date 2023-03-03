from functools import partial
from math import pi as PI

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import Sequential, Linear, BatchNorm1d
from torch_scatter import scatter_add
from torch_geometric.nn import GINConv, GCNConv, global_add_pool, global_mean_pool
from torch.nn import Embedding, Sequential, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.inits import glorot, zeros
from torch_scatter import scatter
from torch_geometric.nn import radius_graph
from typing import Optional, Tuple
from .spherenet import SphereNet

class Encoder(torch.nn.Module):
    
    def __init__(self, args, node_level=False, graph_level=True):
        super(Encoder, self).__init__()
        #self.encoder = args.encoder
        self.cutoff = args.cutoff
        self.num_layers = args.num_layers
        self.hidden_channels = args.z_dim
        
        self.pool = 'sum'
        self.bn = True
        self.act = 'relu'
        self.bias = True
        self.xavier=True
        self.edge_weight=args.edge_weight
        
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
            
        elif args.encoder=='gin':
            self.feat_dim = args.feat_dim
            
            
        elif args.encoder=='gcn':
            self.feat_dim = args.feat_dim
            
        
        self.dropout_rate = args.dropout_rate
        self.device = args.device       
        
        if args.encoder == 'schnet':
            self.encoder = SchNet(cutoff=self.cutoff, num_layers=self.num_layers, hidden_channels=self.hidden_channels, dropout_rate=self.dropout_rate, num_filters=self.num_filters, num_gaussians=self.num_gaussians)
        elif args.encoder == 'spherenet':
            self.encoder = SphereNet(cutoff=self.cutoff, num_layers=self.num_layers, hidden_channels=self.hidden_channels, #out_channels=1, 
                                     int_emb_size=self.int_emb_size, basis_emb_size_dist=self.basis_emb_size_dist, 
                                     basis_emb_size_angle=self.basis_emb_size_angle, basis_emb_size_torsion=self.basis_emb_size_torsion,
                                     out_emb_channels=self.out_emb_channels, num_spherical=self.num_spherical, num_radial=self.num_radial, 
                                     envelope_exponent=self.envelope_exponent, num_before_skip=self.num_before_skip, num_after_skip=self.num_after_skip, 
                                     num_output_layers=self.num_output_layers, dropout_rate=self.dropout_rate, use_node_features=self.use_node_features)
        
        elif args.encoder == 'gin':
            self.encoder = GIN(feat_dim=self.feat_dim, hidden_dim=self.hidden_channels, n_layers=self.num_layers,
                               pool=self.pool, bn=self.bn, act=self.act)
        elif args.encoder == 'gcn':
            self.encoder = GCN(feat_dim=self.feat_dim, hidden_dim=self.hidden_channels, n_layers=self.num_layers,
                               pool=self.pool, bn=self.bn, act=self.act, bias=self.bias, xavier=self.xavier, edge_weight=self.edge_weight)
            
        self.node_level = node_level
        self.graph_level = graph_level

    def forward(self, data):
        z_g = self.encoder(data)
            
        #else:
         #   z_g, z_n = self.encoder(data)
        
        if self.node_level and self.graph_level:
            return z_g, z_n
        elif self.graph_level:
            return z_g
        else:
            return z_n


class update_e(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters, num_gaussians, cutoff, dropout_rate):
        super(update_e, self).__init__()
        self.cutoff = cutoff
        self.lin = Linear(hidden_channels, num_filters, bias=False)
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            #nn.Dropout(dropout_rate),
            Linear(num_filters, num_filters),
        )

        self.reset_parameters()
        self.dropout = nn.Dropout(dropout_rate)
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)

    def forward(self, v, dist, dist_emb, edge_index):
        j, _ = edge_index
        C = 0.5 * (torch.cos(dist * PI / self.cutoff) + 1.0)
        W = self.mlp(dist_emb) * C.view(-1, 1)
        W = self.dropout(W)
        v = self.lin(v)
        e = v[j] * W
        return e


class update_v(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters, dropout_rate):
        super(update_v, self).__init__()
        self.act = ShiftedSoftplus()
        self.lin1 = Linear(num_filters, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()
        self.dropout = nn.Dropout(dropout_rate)
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, v, e, edge_index):
        _, i = edge_index
        out = scatter(e, i, dim=0)
        out = self.lin1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.lin2(out)
        return v + out


class update_u(torch.nn.Module):
    def __init__(self, hidden_channels, dropout_rate):
        super(update_u, self).__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, hidden_channels)

        self.reset_parameters()
        self.dropout = nn.Dropout(dropout_rate)
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, v, batch):
        v = self.lin1(v)
        v = self.act(v)
        v = self.dropout(v)
        v = self.lin2(v)
        u = scatter(v, batch, dim=0)  # 
        return u


class emb(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(emb, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class SchNet(torch.nn.Module):
    r"""
        The re-implementation for SchNet from the `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper
        under the 3DGN gramework from `"Spherical Message Passing for 3D Graph Networks" <https://arxiv.org/abs/2102.05013v2>`_ paper.
        
        Args:
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the negative of the derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)
            num_layers (int, optional): The number of layers. (default: :obj:`6`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
            num_filters (int, optional): The number of filters to use. (default: :obj:`128`)
            num_gaussians (int, optional): The number of gaussians :math:`\mu`. (default: :obj:`50`)
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`10.0`).
    """
    def __init__(self, num_layers, cutoff, hidden_channels, dropout_rate, num_filters=128, num_gaussians=50):
        super(SchNet, self).__init__()

        self.cutoff = cutoff
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians

        self.init_v = Embedding(100, hidden_channels)
        self.dist_emb = emb(0.0, cutoff, num_gaussians)

        self.update_vs = torch.nn.ModuleList([update_v(hidden_channels, num_filters, dropout_rate) for _ in range(num_layers)])

        self.update_es = torch.nn.ModuleList([
            update_e(hidden_channels, num_filters, num_gaussians, cutoff, dropout_rate) for _ in range(num_layers)])
        
        self.update_u = update_u(hidden_channels, dropout_rate)

        self.reset_parameters()

    def reset_parameters(self):
        self.init_v.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()
        self.update_u.reset_parameters()

    def forward(self, batch_data):
        z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=100)
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        dist_emb = self.dist_emb(dist)
        v = self.init_v(z)

        for update_e, update_v in zip(self.update_es, self.update_vs):
            e = update_e(v, dist, dist_emb, edge_index)
            v = update_v(v, e, edge_index)
        u = self.update_u(v, batch)
        return u
    
    
class GIN(torch.nn.Module):
    def __init__(self, feat_dim, hidden_dim, n_layers=3, pool='sum', bn=False, 
                 act='relu', bias=True, xavier=True):
        super(GIN, self).__init__()

        if bn:
            self.bns = torch.nn.ModuleList()
        else:
            self.bns = None
        self.convs = torch.nn.ModuleList()
        self.n_layers = n_layers
        self.pool = pool

        self.act = torch.nn.PReLU() if act == 'prelu' else torch.nn.ReLU()

        for i in range(n_layers):
            start_dim = hidden_dim if i else feat_dim
            nn = Sequential(Linear(start_dim, hidden_dim, bias=bias),
                            self.act,
                            Linear(hidden_dim, hidden_dim, bias=bias))
            if xavier:
                self.weights_init(nn)
            conv = GINConv(nn)
            self.convs.append(conv)
            if bn:
                self.bns.append(BatchNorm1d(hidden_dim))

    def weights_init(self, module):
        for m in module.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()
        xs = []
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            x = self.act(x)

            if self.bns is not None:
                x = self.bns[i](x)
            xs.append(x)

        if self.pool == 'sum':
            xpool = [global_add_pool(x, batch) for x in xs]
        else:
            xpool = [global_mean_pool(x, batch) for x in xs]
        global_rep = torch.cat(xpool, 1)
        return global_rep


class GCN(torch.nn.Module):
    def __init__(self, feat_dim, hidden_dim, n_layers=3, pool='sum', bn=False, 
                 act='relu', bias=True, xavier=True, edge_weight=False):
        super(GCN, self).__init__()

        if bn:
            self.bns = torch.nn.ModuleList()
        else:
            self.bns = None
        self.convs = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        self.n_layers = n_layers
        self.pool = pool
        self.edge_weight = edge_weight
        self.normalize = not edge_weight
        self.add_self_loops = not edge_weight

        if act == 'prelu':
            a = torch.nn.PReLU()
        else:
            a = torch.nn.ReLU()

        for i in range(n_layers):
            start_dim = hidden_dim if i else feat_dim
            conv = GCNConv(start_dim, hidden_dim, bias=bias,
                           add_self_loops=self.add_self_loops,
                           normalize=self.normalize)
            if xavier:
                self.weights_init(conv)
            self.convs.append(conv)
            self.acts.append(a)
            if bn:
                self.bns.append(BatchNorm1d(hidden_dim))

    def weights_init(self, m):
        if isinstance(m, GCNConv):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.edge_weight:
            edge_attr = data.edge_attr
        else:
            edge_attr = None
        xs = []
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.acts[i](x)
            if self.bns is not None:
                x = self.bns[i](x)
            xs.append(x)

        if self.pool == 'sum':
            xpool = [global_add_pool(x, batch) for x in xs]
        else:
            xpool = [global_mean_pool(x, batch) for x in xs]
        global_rep = torch.cat(xpool, 1)

        return global_rep