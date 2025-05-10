import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_logits(target, mask):
    #return target * mask + (1 - mask) * (0)
    return target * mask + (1 - mask) * (-1e30)
def mask_logits_result(target, mask):
    return target * mask + (1 - mask) * (0)

class Rel_GAT(nn.Module):
    """
    Relation gat model, use the embedding of the edges to predict attention weight
    """

    def __init__(self,  dep_rel_num=46, args=None, hidden_size=768,  num_layers=2):
        super(Rel_GAT, self).__init__()
        self.args = args
        self.num_layers = num_layers
        self.dropout = nn.Dropout(0.2)
        self.leakyrelu = nn.LeakyReLU(1e-2)


        # gat layer
        # relation embedding, careful initialization?
        self.dep_rel_embed = nn.Embedding(
            46, 768)
        nn.init.xavier_uniform_(self.dep_rel_embed.weight)

        # map rel_emb to logits. Naive attention on relations
        layers = [
            nn.Linear(768, hidden_size), nn.ReLU(),   #edge_dim, H
            nn.Linear(hidden_size, 1)]
        self.fcs = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(768, eps=1e-6)

    def forward(self, adj, rel_adj, feature):
        denom = adj.sum(2).unsqueeze(2) + 1
        B, N = adj.size(0), adj.size(1)

        # rel_adj_V = self.dep_rel_embed(
        #     rel_adj.view(B, -1))  # (batch_size, n*n, d)
        rel_adj_V = rel_adj.reshape(B,N,N,-1)

        # gcn layer
        for l in range(self.num_layers):
            # relation based GAT, attention over relations

            if True:
                #rel_adj_V=self.layer_norm(rel_adj_V)
                rel_adj_logits = self.fcs(rel_adj_V).squeeze(3)  # (batch_size, n,n)
            else:
                rel_adj_logits = self.A[l](rel_adj_V).squeeze(2)  # (batch_size, n*n)

            dmask = adj.reshape(B,N, N)  # (batch_size, n,n)
            masked_logits=mask_logits(rel_adj_logits, dmask)
            rel_adj_logits = F.softmax(masked_logits, dim=1)
            rel_adj_logits = rel_adj_logits.view(
                *adj.size())  # (batch_size, n, n)
            # rel_adj_logits=mask_logits_result(rel_adj_logits,dmask)

            Ax = torch.bmm(rel_adj_logits,feature)
            feature = self.dropout(Ax) if l < self.num_layers - 1 else Ax

        return feature


class GAT(nn.Module):
    """
    GAT module operated on graphs
    """

    def __init__(self, args, in_dim, hidden_size=64, mem_dim=300, num_layers=2):
        super(GAT, self).__init__()
        self.args = args
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.dropout = nn.Dropout(args.gcn_dropout)
        self.leakyrelu = nn.LeakyReLU(1e-2)

        # Standard GAT:attention over feature
        a_layers = [
            nn.Linear(2 * mem_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)]
        self.afcs = nn.Sequential(*a_layers)

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(num_layers):
            input_dim = self.in_dim if layer == 0 else mem_dim
            self.W.append(nn.Linear(input_dim, mem_dim))

    def forward(self, adj,  feature):
        B, N = adj.size(0), adj.size(1)
        dmask = adj.view(B, -1)  # (batch_size, n*n)
        # gcn layer
        for l in range(self.num_layers):
            # Standard GAT:attention over feature
            #####################################
            h = self.W[l](feature) # (B, N, D)
            a_input = torch.cat([h.repeat(1, 1, N).view(
                B, N*N, -1), h.repeat(1, N, 1)], dim=2)  # (B, N*N, 2*D)
            e = self.leakyrelu(self.afcs(a_input)).squeeze(2)  # (B, N*N)
            attention = F.softmax(mask_logits(e, dmask), dim=1)
            attention = attention.view(*adj.size())

            # original gat
            feature = attention.bmm(h)
            feature = self.dropout(feature) if l < self.num_layers - 1 else feature
            #####################################

        return feature


class GCN(nn.Module):
    """ 
    GCN module operated on graphs
    """

    def __init__(self, args, in_dim, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.args = args
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(args.gcn_dropout)
        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(num_layers):
            input_dim = self.in_dim if layer == 0 else mem_dim
            self.W.append(nn.Linear(input_dim, mem_dim))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def forward(self, adj, feature):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        for l in range(self.num_layers):
            Ax = adj.bmm(feature)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](feature)  # self loop
            AxW /= denom

            # gAxW = F.relu(AxW)
            gAxW = AxW
            feature = self.dropout(gAxW) if l < self.num_layers - 1 else gAxW
        return feature, mask


class RGAT(nn.Module):

    def __init__(self, hidden_size=768, dep_dim=46, num_layers=2, gcn_dropout=0.2):
        super(RGAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(gcn_dropout)
        self.dep_dim = dep_dim
        # gat layer
        self.W = nn.ModuleDict()
        self.lns = nn.ModuleList()
        self.relu = nn.ReLU(inplace=True)

        for layer in range(num_layers):
            self.lns.append(nn.LayerNorm(hidden_size, elementwise_affine=False))
            self.W[str(layer)] = nn.ModuleList()
            for _ in range(dep_dim):
                self.W[str(layer)].append(nn.Linear(hidden_size, hidden_size, bias=False))


    def forward(self, adj, feature,dmask):
        # dmask=BNN,adj=BNNH

        adj=adj.permute(0,3,1,2)
        # adj: [b, whole_tag_size, l, l]  feature: [b, l, h]
        for layer in range(self.num_layers):
            addition = torch.zeros_like(feature)
            for rel in range(self.dep_dim):
                h_r = self.W[str(layer)][rel](feature)
                rel_adj_logits = F.softmax(
                    mask_logits(adj[:, rel, :, :].squeeze(1), dmask), dim=2)
                h_r = self.relu(rel_adj_logits.bmm(h_r))
                addition = addition + h_r

            feature = self.lns[layer](self.relu(addition) / self.dep_dim)
            feature = self.dropout(feature) if layer < self.num_layers - 1 else feature
        return feature       #N  L  H