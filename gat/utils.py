import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

from config import DATA_DIR


def load_data():
    edges = pd.read_csv(f"{DATA_DIR}/edge.csv", header=None)
    feat_train = pd.read_csv(f"{DATA_DIR}/node_feat_train.csv")
    feat_val = pd.read_csv(f"{DATA_DIR}/node_feat_valid.csv")
    feat_test = pd.read_csv(f"{DATA_DIR}/node_feat_test.csv")
    label_train = pd.read_csv(f"{DATA_DIR}/node_label_train.csv")
    label_val = pd.read_csv(f"{DATA_DIR}/node_label_valid.csv")
    
    edge_index = torch.tensor(edges.values.T, dtype=torch.long)
    features = pd.concat([feat_train, feat_val, feat_test], axis=0).sort_values('id').drop(columns=['id']).values
    labels = pd.concat([label_train, label_val, feat_test[['id']]], axis=0).sort_values('id').fillna(-1)['0'].values
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    idx_train = torch.LongTensor(feat_train['id'].values)
    idx_val = torch.LongTensor(feat_val['id'].values)
    idx_test = torch.LongTensor(feat_test['id'].values)

    data = Data(x=features, edge_index=edge_index, y=labels)
    data.idx_train = idx_train
    data.idx_val = idx_val
    data.idx_test = idx_test

    return data


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


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)