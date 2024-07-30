import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .model import GAT
from .utils import load_data


# Inference settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA inference.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--model_path', type=str, default='../models/gat.pth',
                    help='Path to the saved model.')
parser.add_argument('--output_path', type=str, default='../predictions/pred_gat.csv',
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

# Load data
data = load_data()
edge_index, features, labels = data.edge_index, data.x, data.y
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

# Load model
model = GAT(nfeat=features.shape[1],
            nhid=args.hidden,  # This should match the hidden units used during training
            nclass=labels.max().item() + 1,
            dropout=args.dropout)  # This should match the dropout rate used during training

model.load_state_dict(torch.load(args.model_path))
model.eval()

if args.cuda:
    model.cuda()
    data.cuda()

# Inference
output = model(features, edge_index)

# Get the predicted labels
_, pred = torch.max(output[idx_test], dim=1)

# Save to CSV
results = pd.DataFrame({
    'id': idx_test.cpu().numpy(),
    'label': pred.cpu().numpy()
})
results.to_csv(args.output_path, index=False)
print(f"Prediction saved to {args.output_path}")