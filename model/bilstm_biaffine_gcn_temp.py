import math
from typing import Optional

import numpy as np
import torch
from torch.nn import Sequential
from torch_geometric.nn import GCNConv
from .modeling_bart import BartEncoder, BartDecoder, BartModel
from transformers import BartTokenizer
from fastNLP import seq_len_to_mask
from fastNLP.modules import Seq2SeqEncoder, Seq2SeqDecoder, State
import torch.nn.functional as F
# from fastNLP.models import Seq2SeqModel
from .seq2seq_model import Seq2SeqModel
from torch import nn
import random
from transformers.modeling_bart import BaseModelOutput
from .model_gcn import Rel_GAT,RGAT
from .RGAT import RGATEncod

from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Biaffine(nn.Module):
    def __init__(self,  in1_features, in2_features, out_features, bias=(True, True)):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = torch.nn.Linear(in_features=self.linear_input_size,
                                    out_features=self.linear_output_size,
                                    bias=False)

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = torch.ones(batch_size, len1, 1).to('cuda')
            input1 = torch.cat((input1, ones), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = torch.ones(batch_size, len2, 1).to('cuda')
            input2 = torch.cat((input2, ones), dim=2)
            dim2 += 1
        affine = self.linear(input1)
        affine = affine.view(batch_size, len1*self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)
        biaffine = torch.bmm(affine, input2)
        biaffine = torch.transpose(biaffine, 1, 2)
        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)
        return biaffine


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, device, gcn_dim, edge_dim, dep_embed_dim, pooling='avg'):
        super(GraphConvLayer, self).__init__()
        self.gcn_dim = gcn_dim
        self.edge_dim = edge_dim
        self.dep_embed_dim = dep_embed_dim
        self.device = device
        self.pooling = pooling
        self.layernorm = LayerNorm(self.gcn_dim)
        self.W = nn.Linear(self.gcn_dim, self.gcn_dim)

    def forward(self, weight_prob_softmax, weight_adj, gcn_inputs):
        batch, seq, dim = gcn_inputs.shape  # 16,102,300
        weight_prob_softmax = weight_prob_softmax.permute(0, 3, 1, 2)

        gcn_inputs = gcn_inputs.unsqueeze(1).expand(batch, 1, seq, dim)  # 16,50,102,300

        # R(cat) * H(cat)  [16，50，102，102] * [16，50，102，300] = [16,50,102,300]
        Ax = torch.matmul(weight_prob_softmax, gcn_inputs)
        if self.pooling == 'avg':
            Ax = Ax.mean(dim=1)  # pooling   50--》1   16，102，300
        elif self.pooling == 'max':
            Ax, _ = Ax.max(dim=1)
        elif self.pooling == 'sum':
            Ax = Ax.sum(dim=1)
        # Ax: [batch, seq, dim]
        gcn_outputs = self.W(Ax)  # 16*102*300
        # gcn_outputs = self.layernorm(gcn_outputs)  # 16*102*300
        # weights_gcn_outputs = F.elu(gcn_outputs)  # 16*102*300

        node_outputs = gcn_outputs
        # weight_prob_softmax = weight_prob_softmax.permute(0, 2, 3, 1).contiguous()
        # node_outputs1 = node_outputs.unsqueeze(1).expand(batch, seq, seq, dim)  # 行全是node1 的钜阵
        # node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous()  # 列全是node2的钜阵
        edge_outputs = weight_adj

        return node_outputs, edge_outputs
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.bmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

def initializer_1d(input_tensor, initializer):
    assert len(input_tensor.size()) == 1
    input_tensor = input_tensor.view(-1, 1)
    input_tensor = initializer(input_tensor)
    return input_tensor.view(-1)
class MLPRepScorer(nn.Module):
    def __init__(self, input_size, inner_size, output_size, dropout=0.0):
        super(MLPRepScorer, self).__init__()
        self.rep_layer = nn.Linear(input_size, inner_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.scorer = nn.Linear(inner_size, output_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.rep_layer.weight)
        initializer_1d(self.rep_layer.bias, nn.init.xavier_uniform_)
        nn.init.xavier_uniform_(self.scorer.weight)
        initializer_1d(self.scorer.bias, nn.init.xavier_uniform_)

    def forward(self, x):
        rep = self.dropout_layer(
            F.relu(self.rep_layer.forward(x))
        )
        scores = self.scorer.forward(rep)
        return scores
class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()

'''
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    图注意力层
    input: (B,N,C_in)
    output: (B,N,C_out)
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征数
        self.out_features = out_features  # 节点表示向量的输出特征数
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活

        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # 初始化

        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        """
        inp: input_fea [B,N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵  [N, N] 非零即一，数据结构基本知识
        """
        h = torch.matmul(inp, self.W)  # [B, N, out_features]
        N = h.size()[1]  # N 图的节点数

        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features), h.repeat(1, N, 1)], dim=-1).view(-1,
                                                                                                                    N,
                                                                                                                    N,
                                                                                                                    2 * self.out_features)
        # [B, N, N, 2*out_features]

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        # [B, N, N, 1] => [B, N, N] 图注意力的相关系数（未归一化）

        zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷

        attention = torch.where(adj > 0, e, zero_vec)  # [B, N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=2)  # softmax形状保持不变 [B, N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [B, N, N].[B, N, out_features] => [B, N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.relu(h_prime)
            # return h_prime
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
#'''

#'''
class GraphAttentionLayer(nn.Module):
    def __init__(self,in_feature,out_feature,dropout,alpha,concat=True):
        super(GraphAttentionLayer,self).__init__()
        self.in_feature=in_feature
        self.out_feature=out_feature
        self.dropout=dropout
        self.alpha=alpha
        self.concat=concat
        self.pos_embed_dim=1


        self.Wlinear=nn.Linear(in_feature,out_feature)
        # self.W=nn.Parameter(torch.empty(size=(batch_size,in_feature,out_feature)))
        nn.init.xavier_uniform_(self.Wlinear.weight,gain=1.414)

        # self.aiLinear=nn.Linear(out_feature,1)
        # self.ajLinear=nn.Linear(out_feature,1)
        self.aiLinear=nn.Linear(out_feature,self.pos_embed_dim)
        self.ajLinear=nn.Linear(out_feature,self.pos_embed_dim)
        # self.a=nn.Parameter(torch.empty(size=(batch_size,2*out_feature,1)))
        nn.init.xavier_uniform_(self.aiLinear.weight,gain=1.414)
        nn.init.xavier_uniform_(self.ajLinear.weight,gain=1.414)

        self.leakyRelu=nn.LeakyReLU(self.alpha)
        self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
            512,
            self.pos_embed_dim,
        )
        self.wq = MLPRepScorer(out_feature, (out_feature + self.pos_embed_dim) // 2, self.pos_embed_dim)
        self.wk = MLPRepScorer(out_feature, (out_feature + self.pos_embed_dim) // 2, self.pos_embed_dim)
        self.attention=nn.MultiheadAttention(768,8)

    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer):
        # https://kexue.fm/archives/8265
        # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_query_layer = torch.stack(
            [-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1
        ).reshape_as(query_layer)
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos

        return query_layer


    # def getAttentionE(self,Wh):
    #     #重点改了这个函数
    #
    #     Wh1=self.wq(Wh).view(Wh.shape[0], -1, 1, self.pos_embed_dim).transpose(1, 2)
    #     Wh2=self.wk(Wh).view(Wh.shape[0], -1, 1, self.pos_embed_dim).transpose(1, 2)
    #     # Wh1=self.aiLinear(Wh).view(Wh.shape[0], -1, 1, self.pos_embed_dim).transpose(1, 2)
    #     # Wh2=self.ajLinear(Wh).view(Wh.shape[0], -1, 1, self.pos_embed_dim).transpose(1, 2)
    #
    #     sinusoidal_pos = self.embed_positions(Wh.shape[:-1])[
    #                      None, None, :, :
    #                      ]
    #     pWh1 = self.apply_rotary_position_embeddings(sinusoidal_pos, Wh1)
    #     pWh2 = self.apply_rotary_position_embeddings(sinusoidal_pos, Wh2)
    #
    #
    #     # e=Wh1+Wh2   #broadcast add, => e:size(node,node)
    #     # e=torch.matmul(Wh1, Wh2.transpose(-1, -2))
    #     e = torch.matmul(pWh1, pWh2.transpose(-1, -2))
    #     e=e.squeeze(1)
    #     return e


    def getAttentionE(self,Wh):
        #重点改了这个函数
        Wh1=self.aiLinear(Wh)
        Wh2=self.ajLinear(Wh)
        Wh2=Wh2.view(Wh2.shape[0],Wh2.shape[2],Wh2.shape[1])
        # Wh1=torch.bmm(Wh,self.a[:,:self.out_feature,:])    #Wh:size(node,out_feature),a[:out_eature,:]:size(out_feature,1) => Wh1:size(node,1)
        # Wh2=torch.bmm(Wh,self.a[:,self.out_feature:,:])    #Wh:size(node,out_feature),a[out_eature:,:]:size(out_feature,1) => Wh2:size(node,1)

        e=Wh1+Wh2   #broadcast add, => e:size(node,node)
        return self.leakyRelu(e)

    def forward(self,h,adj):
        # print(h.shape)
        Wh=self.Wlinear(h)
        # Wh=torch.bmm(h,self.W)   #h:size(node,in_feature),W:size(in_feature,out_feature) => Wh:size(node,out_feature)
        e=self.getAttentionE(Wh)
        # e,_=self.attention(Wh,Wh,Wh)

        zero_vec=-1e9*torch.ones_like(e)
        attention=torch.where(adj>0,e,zero_vec)
        attention=F.softmax(attention,dim=2)
        attention=F.dropout(attention,self.dropout,training=self.training)
        h_hat=torch.bmm(attention,Wh)  #attention:size(node,node),Wh:size(node,out_fature) => h_hat:size(node,out_feature)

        if self.concat:
            return F.elu(h_hat)
            # return h_hat
        else:
            return h_hat

    def __repr__(self):
        return self.__class__.__name__+' ('+str(self.in_feature)+'->'+str(self.out_feature)+')'
# '''

class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads):
        """Dense version of GAT
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(GAT, self).__init__()
        self.dropout = dropout
        self.linear=nn.Linear(768,768)

        # 定义multi-head的图注意力层
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)  # 将每个head得到的表示进行拼接
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = self.out_att(x, adj)
        # print(x.shape)
        # x = F.elu(self.out_att(x, adj))  # 输出并激活
        # return F.log_softmax(x, dim=2)  # log_softmax速度变快，保持数值稳定
        # return self.linear(x)
        return x


import seaborn as sns
import matplotlib.pyplot as plt
def show_heatmap(adj):
    batch, maxlen, _, edge_dim = adj.shape
    for i in range(batch):
        adj_new = adj[i].reshape((maxlen, maxlen))
        sns.heatmap(adj_new.cpu().detach().numpy())
        # 不要坐标轴刻度
        plt.gca().xaxis.set_major_locator(plt.AutoLocator())
        plt.gca().yaxis.set_major_locator(plt.AutoLocator())
        # 不要边缘太多的空白
        # plt.tight_layout()
        # 进一步减少边缘
        # plt.savefig('seaborn.jpg', bbox_inches='tight', pad_inches=0)
        # save在show之前
        plt.show()





class biGRU_biaffine_gcn(nn.Module):
    def __init__(self,embedding_dim):
        super().__init__()
        adjacency_dim=1
        self.adjacency_dim=adjacency_dim
        self.embedding_dim=embedding_dim
        self.biGRU = nn.GRU(self.embedding_dim, self.embedding_dim//2, num_layers=1, batch_first=True, bidirectional=True)
        self.triplet_biaffine = Biaffine(self.embedding_dim, self.embedding_dim, adjacency_dim, bias=(True, True))
        self.ap_fc = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.op_fc = nn.Linear(self.embedding_dim, self.embedding_dim)  # 768,300

        self.dense = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.num_layers = 2
        self.gcn_layers = nn.ModuleList()
        self.GAT=GAT(self.embedding_dim, self.embedding_dim,self.embedding_dim,dropout=0.2,alpha=0.01,n_heads=8)
        self.Rel_GAT = Rel_GAT()                                         # from RGAT in ABSA
        self.RGATEnc = RGATEncod(num_layers=2, d_model=768, heads=8, d_ff=256, dropout=0.2, att_drop=0.2,
                                 use_structure=True,
                                 dep_dim=768, alpha=0, beta=1, )
        self.layernorm = LayerNorm(self.embedding_dim)

        for i in range(self.num_layers):
            self.gcn_layers.append(
                GraphConvLayer('cuda', self.embedding_dim, adjacency_dim, adjacency_dim, pooling='avg')
                #GraphConvolution(self.embedding_dim,self.embedding_dim)
            )

        self.MLP = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim),
                                 nn.ELU(),
                                 nn.Linear(self.embedding_dim, self.embedding_dim),
                                 # nn.ReLU(),
                               )

    def forward(self, sentence, batch_size, src_seq_len, head,senti_value,pos_mask,head_len,matrix_mask=None,pos_attention_mask=None,deprel_emb=None):  # sentence是已经编码的句子
        # BiLSTM
        '''
        hidden_state = torch.rand(2, batch_size, 768//2).to('cuda')
        embed_input_x_packed = pack_padded_sequence(sentence, src_seq_len.to('cpu'), batch_first=True,
                                                    enforce_sorted=False)
        lstm_feature, (_, _) = self.biGRU(embed_input_x_packed, hidden_state)    # suppose to be 768

        # out_pad, out_len = pad_packed_sequence(lstm_feature, batch_first=True, padding_value=-100)
        out_pad, out_len = pad_packed_sequence(lstm_feature, batch_first=True,)
        lstm_feature = out_pad
        '''
        lstm_feature = sentence
        # lstm_feature, last_hidden = self.biGRU(sentence)
        # BiAffine
        # ap_node = F.relu(self.ap_fc(lstm_feature))
        # op_node = F.relu(self.op_fc(lstm_feature))
        # biaffine_edge = self.triplet_biaffine(ap_node, op_node)


        #========================================================================here has been modified======================================
        # biaffine_edge_softmax = F.softmax(biaffine_edge, dim=-1)
        # biaffine_edge_relu = F.relu(biaffine_edge, inplace=True)
        # biaffine_edge_relu=_get_graph(head, head_len, batch_size, self.adjacency_dim)
        biaffine_edge_relu, matrix_mask = _get_graph_cut(head, head_len, batch_size, self.adjacency_dim, matrix_mask)
       # show_heatmap(biaffine_edge_relu)
        biaffine_edge=None
        gcn_outputs = lstm_feature
        weight_prob = biaffine_edge
        #gcn_outputs=self.GAT(gcn_outputs,matrix_mask)
        # gcn_outputs=self.Rel_GAT(matrix_mask,biaffine_edge_relu,gcn_outputs)

        bool_tensor = torch.eq(matrix_mask, 0)
        # bool_tensor = torch.eq(biaffine_edge_relu, 0)
        gcn_outputs=self.RGATEnc(gcn_outputs,mask=bool_tensor,   structure=biaffine_edge_relu,
        src_key_padding_mask=pos_attention_mask.unsqueeze(dim=1))
        #for _layer in range(self.num_layers):
        #     H 节点表示 , final_pred 矩阵graphgraph
        #    gcn_outputs, weight_prob = self.gcn_layers[_layer](biaffine_edge_relu, weight_prob, gcn_outputs)  # [batch, seq, dim]
        #     gcn_outputs = self.gcn_layers[_layer](gcn_outputs, biaffine_edge_relu)  # [batch, seq, dim]
            # weight_prob_list.append(weight_prob)
        # gcn_outputs = self.MLP(gcn_outputs)

        res = []
        # 同一个word可能被分为不同的token，因此根据pos_mask将部分embedding进行重复
        for i in range(batch_size):
            temp_pos_mask = pos_mask[
                                i] - 1  # pos_mask从bos开始数，第一个位置是bos，所以统一减1，正好变成[-1,0,1,2..., -2,-2]，其中-2是-1padding减1的结果
            pos = gcn_outputs[i]
            trans_mask = torch.zeros((temp_pos_mask.size()[-1], pos.size()[0])).to(device=pos.device)
            # 利用矩阵乘法拷贝被分为多个token的word_pos
            for j in range(len(temp_pos_mask)):
                if temp_pos_mask[j] >= 0:
                    trans_mask[j, temp_pos_mask[j]] = 1.0
            new_pos = torch.matmul(trans_mask, pos)
            # pad到同样的维度，以方便后续stack
            res.append(new_pos)
        gcn_outputs = torch.stack(res, dim=0)
        return gcn_outputs, biaffine_edge_relu
def _get_graph(head, srcseqlen, batch_size, adjacency_dim):
    maxlen=max(srcseqlen)
    graph_final=torch.zeros(head.shape[0],maxlen,maxlen,adjacency_dim).to("cuda")
    for i in range(batch_size):
        headi=head[i][:srcseqlen[i]]
        # assert srcseqlen[i]-2-3==len(headi)
        # graph=torch.eye(srcseqlen[i]-2).to("cuda")
        # graph=torch.zeros(maxlen,maxlen).to('cuda')
        graph=torch.eye(maxlen).to("cuda")
        for j in range(len(headi)):
            if j == headi[j] :
                continue
            graph[j][headi[j]]+=1
            graph[headi[j]][j]+=1
        # padding = (0,0,1, maxlen-len(headi)-1, 1, maxlen-len(headi)-1)
        # padded_tensor = nn.functional.pad(graph, padding, mode='constant',value=0).to('cuda')
        # graph_final[i]=padded_tensor
        graph = graph.reshape(maxlen, maxlen, 1)
        graph_final[i]=graph
    #return padded_tensor
    # 加 t
    return graph_final.squeeze(dim=3)
    # return graph_final
def _get_graph_cut(head, head_len, batch_size, adjacency_dim,matrix_mask=None):
    maxlen=max(head_len)
    graph=head[:,1:maxlen+1,1:maxlen+1,:]
    matrix_mask=matrix_mask[:,1:maxlen+1,1:maxlen+1]
    return graph,matrix_mask

def invert_mask(attention_mask):
    """Turns 1->0, 0->1, False->True, True-> False"""
    assert attention_mask.dim() == 2
    return attention_mask.eq(0)


# Copied from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding with Marian->RoFormer
class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(
            self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)



class DualBartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:EncoderLayer.

    Args:
        config: MultiModalBartConfig
    """
    def __init__(self,  encoder, tokenizer):
        super().__init__()
        self.use_dual_encoder=True
        self.tokenizer=tokenizer
        embed_tokens = encoder.embed_tokens
        self.dropout = encoder.dropout
        self.layerdrop = encoder.layerdrop
        self.indentity = nn.Identity()

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = encoder.embed_scale
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = encoder.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_images = biGRU_biaffine_gcn(embed_dim)
        self.embed_positions = encoder.embed_positions

        self.layers = encoder.layers
        self.layernorm_embedding = encoder.layernorm_embedding
        # mbart has one extra layer_norm
        self.layer_norm = encoder.layer_norm
        # for p in self.parameters():
        #     print(type(p), p.shape)
        self.nonlinear=nn.Sequential(nn.Linear(embed_dim,1),
                                     nn.Sigmoid()
                                     )
        # 构造词性embedding,46是词性表的长度加一个pad
        self.part_of_speech_embed = nn.Embedding(47, 768, padding_idx=0)  # 46+1
        torch.nn.init.xavier_uniform_(self.part_of_speech_embed.weight)  # 缩小embedding的初始化权重，否则训练效果会很差
        self.deprel_embed = nn.Embedding(46, 768//2, padding_idx=0)  # 45+1
        torch.nn.init.xavier_uniform_(self.deprel_embed.weight)  # 缩小embedding的初始化权重，否则训练效果会很差
        self.pair_position_embed = nn.Embedding(139, 768//4, padding_idx=0)  # 138+1
        torch.nn.init.xavier_uniform_(self.pair_position_embed.weight)  # 缩小embedding的初始化权重，否则训练效果会很差
        self.tree_position_embed = nn.Embedding(8, 768//4, padding_idx=0)  # 7+1
        torch.nn.init.xavier_uniform_(self.tree_position_embed.weight)  # 缩小embedding的初始化权重，否则训练效果会很差


    def _embed_multi_modal(self, input_ids, syntactic, src_seq_len, head,senti_value,pos_mask,head_len,
                           word_pair_deprel=None, matrix_mask=None, pos_attention_mask=None,deprel_ids=None,tree_position=None,pair_position=None):
        """embed textual and visual inputs and combine them into one embedding"""
        # mask = (input_ids == self.img_feat_id) | (
        #     input_ids == self.cls_token_id)
        # print(mask.shape)
        embedded = self.embed_tokens(input_ids)
        # mask_choose = torch.tensor(syntactic != 1).float()

        #syntactics = self.part_of_speech_embed(syntactic)
        syntactics = self.embed_tokens(syntactic)
        deprel_emb = self.deprel_embed(deprel_ids)  # dep_emb
        dep_matrix = self.deprel_embed(word_pair_deprel)  # dep_matrix emb
        pair_position_matrix = self.pair_position_embed(pair_position.long())  # pair_position_emb
        tree_position_matrix = self.tree_position_embed(tree_position.long())  # tree_position_emb
        dep_matrix = torch.cat([dep_matrix, tree_position_matrix, pair_position_matrix], dim=-1)
        # syntactics[:,0]=torch.zeros(syntactic.shape[0],768).to('cuda')
        # syntactics = torch.einsum('bji,bj->bji',[syntactics,mask_choose])
        syntactics = syntactics * self.embed_scale
        # syntactics = self.layernorm_embedding(syntactics)    #完全没用，加了下降
        # =============================================================================================here to modify
        batch_size = syntactics.shape[0]
        embedded_images, biaffine_edge_relu = self.embed_images(syntactics, batch_size, src_seq_len, dep_matrix,senti_value,pos_mask,head_len,
                                                                matrix_mask=matrix_mask,pos_attention_mask=pos_attention_mask,deprel_emb=deprel_emb)
        # print('mask shape', mask.shape)
        if not embedded_images[0].dtype == torch.float32:
            embedded = embedded.half()

        # for index, value in enumerate(embedded_images):
        #     if len(value) > 0:
        #         embedded[index, mask[index]] = value
        return embedded, biaffine_edge_relu, embedded_images

    def forward(self,
                input_ids,
                input_syntactics,senti_value,
                src_seq_len,
                head=None,word_pair_deprel=None, matrix_mask=None,pos_attention_mask=None,deprel_ids=None,
                word_index=None,head_len=None,tree_position=None,pair_position=None,
                attention_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False):
        """

        :param input_ids: LongTensor, tokens in the source language of shape (batch, src_len)
        :param image_features: list[FloatTensor], image roi features with length of batch
        :param attention_mask: LongTensor, indicating which indices are padding tokens.
        :param output_attentions:
        :param output_hidden_states:
        :return: Tuple comprised of:
            - x (Tensor): the last encoder layer's output of
              shape (src_len, batch, embed_dim)
            - encoder_states (List[Tensor]): all intermediate
              hidden states of shape (src_len, batch, embed_dim).
              Only populated if output_hidden_states: is True.
            - all_attentions (List[Tensor]): Attention weights for each layer.
            During training might not be of length n_layers because of layer dropout.
        """
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)
        if pos_attention_mask is not None:
            pos_attention_mask = invert_mask(pos_attention_mask)

        a=self.tokenizer.convert_ids_to_tokens(torch.tensor([50285]).to('cuda')) #NULL--50285
        b=self.tokenizer.convert_ids_to_tokens(input_ids[0])
        c=self.tokenizer.convert_ids_to_tokens(input_syntactics[0])
        se_embeds, biaffine_edge_relu, sy_embeds = self._embed_multi_modal(input_ids, input_syntactics, src_seq_len, head,senti_value,
                                                                           word_index, head_len,
                                                                           word_pair_deprel=word_pair_deprel,
                                                                           matrix_mask=matrix_mask,
                                                                           pos_attention_mask=pos_attention_mask,
                                                                           deprel_ids=deprel_ids,
                                                                           tree_position=tree_position,
                                                                           pair_position=pair_position
                                                                           )


        se_embeds = se_embeds * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        se_embeds=se_embeds+embed_pos
        # x2=torch.cat((sy_embeds,se_embeds),dim=1)
        x2=sy_embeds
        x3=se_embeds.clone()
        x=se_embeds
        # x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        # x2 = self.layernorm_embedding(x2)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states, all_attentions = [], []
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability <
                                  self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x,
                                        attention_mask,
                                        output_attentions=output_attentions)

            if output_attentions:
                all_attentions.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)

        # T x B x C -> B x T x C
        encoder_states = [
            hidden_state.transpose(0, 1) for hidden_state in encoder_states
        ]
        x = x.transpose(0, 1)

        nonlinear_outputs=self.nonlinear(x)
        if not return_dict:
            return tuple(v for v in [x, x2,x3, all_attentions, biaffine_edge_relu, nonlinear_outputs]
                         if v is not None)
        return BaseModelOutput(last_hidden_state=x,
                               hidden_states=encoder_states,
                               attentions=all_attentions)

