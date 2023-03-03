from math import pi as PI
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear
from torch_scatter import scatter
from torch_geometric.nn import radius_graph, global_add_pool, global_mean_pool


class update_e(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters, num_gaussians, cutoff, dropout_rate):
        super(update_e, self).__init__()
        self.cutoff = cutoff
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.lin = Linear(hidden_channels, num_filters, bias=False)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout_rate)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)

    def forward(self, v, dist, dist_emb, edge_index):
        #print('-------------update e-------------')
        j, _ = edge_index
        #print('j(index of n_j: ', j.shape)
        #print('j[:5]: ',j[:5])
        C = 0.5 * (torch.cos(dist * PI / self.cutoff) + 1.0)       
        #print('C(embedding of dist/cutoff): ', C.shape)
        W = self.mlp(dist_emb) * C.view(-1, 1)
        #print('W(MLP with dist_emb and C): ', W.shape)
        W = self.dropout(W)
        v = self.lin(v)
        #print('v(hidden_channels -> num_filters): ', v.shape)
        e = v[j] * W
        #print('e(v[j] * W : node_j에 해당하는 node만 indexing한 후 W multiply -> j atom에 dist와 dist_emb 반영): ', e.shape)
        #print('dist: ', dist)
        #print('dist_emb: ', dist_emb)
        #print('W: ', W)
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
        #print('-------------update v-------------')
        _, i = edge_index
        #print('i(index of n_i): ', i.shape)
        #print('i[:5]: ',i[:5])
        #print('update e: ', e.shape)  # atom_j 들의 순서가 edge_index에 따라 배열
        #print('e', e[:2])
        #print('e[i]', e[i][:2])
        #print('e[i](node i에 해당하는 update e만 indexing): ', e[i].shape)  # edge_index에 따라 atom_j들이 배열되어 있기 때문에 atom_i의 index로 _j를 가져올 수 있음
        #print('e shape: ', e.shape)
        #print('i(index of n_i): ', i.shape)
        #print('v.shape', v.shape)
        out = scatter(e, i, dim=0)  # 연결되어있는 j들을 sum해서 atom_i 생성
        #out = global_add_pool(e, i)
        #print('out.shape', out.shape)
        #print('out(update e 중에 node i에 해당하는 e들만 sum): ', out.shape)  # 
        out = self.lin1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.lin2(out)
        #print('out(num_filters -> hidden channel MLP에 태움): ', out.shape)
        #print(self.lin2.weight)
        
        #if v.shape != out.shape:
            #print('lin2 weight.shape', self.lin2.weight.shape)
            #print('v', v)
            #print('out', out)
            #print('v[-2]', v[-2])
            #print('v[-1]', v[-1])
            #print('out', out[-1])
            #print('v.shape', v.shape)
            #print('out.shape', out.shape)
        return v + out


class update_u(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout_rate):
        super(update_u, self).__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, out_channels)

        self.reset_parameters()
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

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
        u = scatter(v, batch, dim=0)
        #out = self.sigmoid(u)
        return u


class emb(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(emb, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)  # start 부터 stop까지 일정 간격의 50개의 number 생성
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

    def __init__(self, cutoff, num_layers, hidden_channels, out_channels, num_filters, num_gaussians, dropout_rate):
        super(SchNet, self).__init__()

        self.cutoff = cutoff
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians
        self.dropout_rate = dropout_rate

        self.init_v = Embedding(100, hidden_channels)
        self.dist_emb = emb(0.0, cutoff, num_gaussians)

        self.update_vs = torch.nn.ModuleList([update_v(hidden_channels, num_filters, dropout_rate) for _ in range(num_layers)])

        self.update_es = torch.nn.ModuleList([
            update_e(hidden_channels, num_filters, num_gaussians, cutoff, dropout_rate) for _ in range(num_layers)])
        
        self.update_u = update_u(hidden_channels, out_channels, dropout_rate)

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
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=100)  # Based on pos, return edge_index within cutoff 
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1) # return distance between edge
        dist_emb = self.dist_emb(dist)
        v = self.init_v(z)
        for update_e, update_v in zip(self.update_es, self.update_vs):
            e = update_e(v, dist, dist_emb, edge_index)
            #print('e: ', e.shape)
            v = update_v(v, e, edge_index)
            #print('v: ', v.shape)
        u = self.update_u(v, batch)
        return u
