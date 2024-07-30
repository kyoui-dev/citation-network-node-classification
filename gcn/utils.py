import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch

from config import DATA_DIR


def load_data():
    edges = pd.read_csv(f"{DATA_DIR}/edge.csv", header=None)
    feat_train = pd.read_csv(f"{DATA_DIR}/node_feat_train.csv")
    feat_val = pd.read_csv(f"{DATA_DIR}/node_feat_valid.csv")
    feat_test = pd.read_csv(f"{DATA_DIR}/node_feat_test.csv")
    label_train = pd.read_csv(f"{DATA_DIR}/node_label_train.csv")
    label_val = pd.read_csv(f"{DATA_DIR}/node_label_valid.csv")

    edges = clean_edges(edges)
    features = pd.concat([feat_train, feat_val, feat_test], axis=0).sort_values('id').drop(columns=['id']).values
    labels = pd.concat([label_train, label_val, feat_test[['id']]], axis=0).sort_values('id').fillna(-1)['0'].values
    adj = sp.lil_matrix((features.shape[0], features.shape[0]), dtype=int)
    for edge in edges.values:
        i, j = edge
        adj[j, i] = 1
    adj = normalize((adj + sp.eye(adj.shape[0])).T)
    
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(feat_train['id'].values)
    idx_val = torch.LongTensor(feat_val['id'].values)
    idx_test = torch.LongTensor(feat_test['id'].values)

    return adj, features, labels, idx_train, idx_val, idx_test


def clean_edges(edges):
    year_train = pd.read_csv(f"{DATA_DIR}/node_year_train.csv")
    year_val = pd.read_csv(f"{DATA_DIR}/node_year_valid.csv")
    year_test = pd.read_csv(f"{DATA_DIR}/node_year_test.csv")

    years = pd.concat([year_train, year_val, year_test], axis=0).sort_values('id').values
    id_year_map = {id: year for id, year in years}
    edge_index = edges.values
    mask = np.array([id_year_map[i] >= id_year_map[j] for i, j in edge_index])
    edges = pd.DataFrame(edge_index[mask])

    return edges


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)