from math import pi as PI
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear, BatchNorm1d
from torch_scatter import scatter
from torch_geometric.nn import radius_graph, global_add_pool, global_mean_pool
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import GINConv, GCNConv, global_add_pool, global_mean_pool


class GIN(torch.nn.Module):
    def __init__(self, feat_dim, hidden_dim, out_dim, dropout_rate, n_layers=3, pool='sum', bn=False, 
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
        self.dropout = torch.nn.Dropout(dropout_rate)
        
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
        
        self.ffn = Sequential(Linear(hidden_dim * n_layers, hidden_dim // 2),
                              torch.nn.ReLU(), torch.nn.Dropout(dropout_rate),
                              Linear(hidden_dim // 2, out_dim))
        
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
            x = self.dropout(x)
            if self.bns is not None:
                x = self.bns[i](x)
            xs.append(x)

        if self.pool == 'sum':
            xpool = [global_add_pool(x, batch) for x in xs]
        else:
            xpool = [global_mean_pool(x, batch) for x in xs]
        global_rep = torch.cat(xpool, 1)
        out = self.ffn(global_rep)
        return out