import torch
import dgl
import torch.nn.functional as F
import numpy as np
from utils import map_activation_str_to_layer


class EdgeEmbedding_F3_4(torch.nn.Module):
    def __init__(self, input_channels, output_channels, act_func, bsz):
        super(EdgeEmbedding_F3_4, self).__init__()
        self.input_c = input_channels
        self.output_c = output_channels
        self.act = act_func
        self.bsz = bsz
        self.linear1 = torch.nn.Linear(2*input_channels, output_channels)

    def forward(self, attr, neigh_agg):
        result = self.linear1(torch.cat((attr,neigh_agg),dim=2))
        result = self.act(result)
        return result


class Mean_Agg(torch.nn.Module):
    def __init__(self, in_channels, act_func, dropuout):
        super(Mean_Agg, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.act = act_func
        self.droput = dropuout
        # 用于重置参数

    def forward(self, attr, adj, mask):
        out = torch.matmul(adj, attr)
        zero = torch.count_nonzero(adj, dim=1)
        zero = zero.unsqueeze(2) + 1
        out = out / zero
        out = F.dropout(out, p=self.droput, training=self.training)
        return out


class MeanN(torch.nn.Module):
    def __init__(self, input_channels, output_channels, act_func, dropout, bsz):
        super(MeanN, self).__init__()
        self.gat = Mean_Agg(input_channels, act_func, dropout)
        self.emb = EdgeEmbedding_F3_4(input_channels, output_channels, act_func, bsz)

    def forward(self, attr, adj, p_max=0, mask=None):
        attr_agg = self.gat(attr, adj, p_max, mask)
        emb = self.emb(attr, attr_agg)
        # emb = emb.resize(emb.size(0)*emb.size(1),emb.size(2))
        return emb
