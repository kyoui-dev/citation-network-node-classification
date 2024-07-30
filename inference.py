import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import DATA_DIR, MODEL_DIR
from src.models import GCN, GAT
from utils import load_data

# GCN
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA inference.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--model_path', type=str, default=f"{MODEL_DIR}/gcn.pth",
                    help='Path to the saved model.')
parser.add_argument('--output_path', type=str, default=f"{DATA_DIR}/pred_gcn.csv",
                    help='Path to save the output csv.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

adj, features, labels, idx_train, idx_val, idx_test = load_data()
model = GCN(nfeat=features.shape[1],
            nhid=64,
            nclass=40,
            dropout=0.5)
model.load_state_dict(torch.load(args.model_path))
model.eval()

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    idx_test = idx_test.cuda()

output = model(features, adj)
_, pred = torch.max(output, dim=1)
results = pd.DataFrame({
    'id': idx_test.cpu().numpy(),
    'label': pred[idx_test].cpu().numpy()
})
results.to_csv(args.output_path, index=False)
print(f"Prediction saved to {args.output_path}")

# GAT
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA inference.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--model_path', type=str, default=f"{MODEL_DIR}/gat.pth",
                    help='Path to the saved model.')
parser.add_argument('--output_path', type=str, default=f"{DATA_DIR}/pred_gat.csv",
                    help='Path to save the output csv.')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

data = load_data()
edge_index, features, labels = data.edge_index, data.x, data.y
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
model = GAT(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout) 

model.load_state_dict(torch.load(args.model_path))
model.eval()

if args.cuda:
    model.cuda()
    data.cuda()

output = model(features, edge_index)
_, pred = torch.max(output[idx_test], dim=1)
results = pd.DataFrame({
    'id': idx_test.cpu().numpy(),
    'label': pred.cpu().numpy()
})
results.to_csv(args.output_path, index=False)
print(f"Prediction saved to {args.output_path}")