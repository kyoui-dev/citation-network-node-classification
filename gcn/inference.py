import argparse
import numpy as np
import pandas as pd
import torch

from .model import GCN
from .utils import load_data

from config import MODEL_DIR, OUTPUT_DIR


# Inference settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA inference.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--model_path', type=str, default=f"{MODEL_DIR}/gcn.pth",
                    help='Path to the saved model.')
parser.add_argument('--output_path', type=str, default=f"{OUTPUT_DIR}/pred_gcn.csv",
                    help='Path to save the output csv.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Load model
model = GCN(nfeat=features.shape[1],
            nhid=64,  # This should match the hidden units used during training
            nclass=40,
            dropout=0.5)  # This should match the dropout rate used during training

model.load_state_dict(torch.load(args.model_path))
model.eval()

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    idx_test = idx_test.cuda()

# Inference
output = model(features, adj)

# Get the predicted labels
_, pred = torch.max(output, dim=1)

# Save to CSV
results = pd.DataFrame({
    'id': idx_test.cpu().numpy(),
    'label': pred[idx_test].cpu().numpy()
})
results.to_csv(args.output_path, index=False)
print(f"Prediction saved to {args.output_path}")