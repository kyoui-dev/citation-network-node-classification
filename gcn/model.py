import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid * 2, nclass)
        self.res1 = nn.Linear(nfeat, nhid)
        self.res2 = nn.Linear(nhid * 2, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        # Vanilla model
        # h = F.relu(self.gc1(x, adj))
        # h = F.dropout(h, self.dropout, training=self.training)
        # h = self.gc2(h, adj)
        # h = F.dropout(h, self.dropout, training=self.training)

        # Residual model
        h = torch.concat([F.relu(self.gc1(x, adj)), self.res1(x)], dim=1)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.gc2(h, adj) + self.res2(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return F.log_softmax(h, dim=1)
