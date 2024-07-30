import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.gat1 = GATConv(nfeat, nhid, heads=8)
        self.gat2 = GATConv(nhid * 8, nclass, heads=1)
        self.res1 = nn.Linear(nfeat, nhid * 8)
        self.res2 = nn.Linear(nhid * 8, nclass)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Vanilla model
        # x = F.dropout(x, self.dropout, training=self.training)
        # h = F.elu(self.gat1(x, edge_index)) 
        # h = F.dropout(h, self.dropout, training=self.training)
        # h = self.gat2(h, edge_index)
        
        # Residual model
        h = F.elu(self.gat1(F.dropout(x, self.dropout, training=self.training), edge_index)) + self.res1(x)
        h = self.gat2(F.dropout(h, self.dropout, training=self.training), edge_index) + self.res2(h)
        return F.log_softmax(h, dim=1)