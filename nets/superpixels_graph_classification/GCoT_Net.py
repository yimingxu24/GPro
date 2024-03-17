import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import dgl.function as fn
from layers.mlp_readout_layer import MLPReadout
import pdb
import torch.autograd as autograd
import numpy as np


class GCoTNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.n_ks = net_params['ks']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']

        self.causal_features = nn.Linear(in_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

        self.MLP_layer = nn.Linear(out_dim * 2, n_classes) 
        self.MLP_layer_c = nn.Linear(out_dim, n_classes) 
        self.MLP_layer_b = nn.Linear(out_dim, n_classes) 

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.feat_transform = nn.Linear(hidden_dim, 64)
        self.mlp = MLPReadout(hidden_dim * 2 + 1, 1)

        self.layers = nn.ModuleList([GCoTLayer(hidden_dim, hidden_dim, hidden_dim, F.relu, dropout,
                                              self.n_ks, self.batch_norm, self.residual)])
        self.layers.append(GCN(hidden_dim, hidden_dim, out_dim, F.relu, dropout, self.batch_norm, self.residual))


    def forward(self, g, h, largest):
        h = self.causal_features(h)

        for conv in self.layers[:-1]:
            g, h, link_scores, data_mask = conv(g, h, largest)
        
        if largest == False:
            data_mask = 1 - data_mask

        h = self.layers[-1](g, h, data_mask)


        g.ndata['h'] = h
        g = g.to(torch.device("cuda"))

        hg = dgl.readout_nodes(g, 'h', op=self.readout)

        return hg, link_scores

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss(reduction='none')
        # criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
   

    def test(self, g, h, e, data_mask=None, data_mask_node=None):
        results = self.forward(g, h, None, data_mask, None)
        return results
    

class GCoTLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, activation, dropout, ks, batch_norm, residual=False):
        super().__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.activation = activation
        self.dropout = dropout
        self.ks = ks
        self.batch_norm = batch_norm
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.residual = residual

        self.gcns = nn.ModuleList()
        self.CoTs = nn.ModuleList()

        for i in range(len(self.ks)):
            self.gcns.append(GCN(self.in_dim, self.hid_dim, self.out_dim, self.activation, self.dropout, self.batch_norm, self.residual))
            self.CoTs.append(CoT(ks[i], self.out_dim, self.dropout))



    def forward(self, g, h, largest, data_mask = None):
        features = h
        link_scores, eid_list, eid = [], [], None
        
        for i in range(len(self.ks)):
            h = self.gcns[i](g, h, data_mask)
            g, h, scores, data_mask = self.CoTs[i](g, h, largest)

            link_scores.append(scores)

        return g, h, link_scores, data_mask
    

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, activation, dropout, batch_norm, residual=False):
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        self.batchnorm_h = nn.BatchNorm1d(out_size)

        self.residual = residual

        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu, allow_zero_in_degree=True)
        )
        self.layers.append(dglnn.GraphConv(hid_size, out_size, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features, data_mask=None):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h, edge_weight=data_mask)
        
        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = features + h # residual connection

        if self.batch_norm:
            h = self.batchnorm_h(h) # batch normalization     

        return h



class CoT(nn.Module):

    def __init__(self, k, in_dim, dropout):
        super(CoT, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.dropout = nn.Dropout(dropout)

        self.mlp = MLPReadout(64 * 2 + 1, 1)
        self.feat_transform = nn.Linear(in_dim + 1, 64)


    def forward(self, g, h, largest):
        scores = self.concat_mlp_score(g, h)

        return top_k_edge_graph(scores, g, h, self.k, largest)


    def concat_mlp_score(self, g, h):
        row, col = g.edges()

        row_node_feat = torch.cat(((g.out_degrees(row) + g.in_degrees(row)).unsqueeze(-1), h[row]), dim=1)
        row_node_feat = self.feat_transform(row_node_feat)

        col_node_feat = torch.cat(((g.out_degrees(col) + g.in_degrees(col)).unsqueeze(-1), h[col]), dim=1)
        col_node_feat = self.feat_transform(col_node_feat)

        node_similarity = torch.cosine_similarity(row_node_feat, col_node_feat)

        link_score = torch.cat((node_similarity.unsqueeze(-1), row_node_feat, col_node_feat), dim=1)
        link_score = self.mlp(link_score)
        link_score = self.sigmoid(link_score.squeeze())

        return link_score


def top_k_edge_graph(scores, g, h, k, largest):    
    batch_num_nodes = g.batch_num_nodes()
    batch_num_edges = g.batch_num_edges()

    edge_idx_list, node_num_list, edge_num_list= [], [], []
    edge_mask = torch.tensor([]).to(torch.device("cuda"))

    for idx in range(g.batch_size):
        start_edge = torch.sum(batch_num_edges[:idx])
        end_edge = torch.sum(batch_num_edges[:idx+1])

        values, idxs = torch.topk(scores[start_edge: end_edge], max(2, int(batch_num_edges[idx] * k)), largest=largest)

        edge_idx = (idxs + start_edge.item()).tolist()
        edge_idx_list.extend(edge_idx)
        edge_mask = torch.cat([edge_mask, values])

        edge_num_list.append(idxs.shape[0])

    sg = dgl.edge_subgraph(g, edge_idx_list, relabel_nodes=False)
    sg.set_batch_num_nodes(torch.tensor(batch_num_nodes))
    sg.set_batch_num_edges(torch.tensor(edge_num_list))

    return sg, h, scores, edge_mask.unsqueeze(1)