import time

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

from .model import GAT
from .utils import load_data, accuracy


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--model_path', type=str, default="../models/gat.pth")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

wandb.init(project="gat", config=args)
config = wandb.config

wandb.config.update({
    "learning_rate": args.lr,
    "epochs": args.epochs,
    "weight_decay": args.weight_decay,
    "hidden_units": args.hidden,
    "dropout": args.dropout,
    "seed": args.seed
})

# Load data
data = load_data()
edge_index, features, labels = data.edge_index, data.x, data.y
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

# Model and optimizer
model = GAT(nfeat=data.num_features,
            nhid=args.hidden,
            nclass=data.y.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    data.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, edge_index)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, edge_index)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print("Epoch: {:04d}".format(epoch+1),
          "loss_train: {:.4f}".format(loss_train.item()),
          "acc_train: {:.4f}".format(acc_train.item()),
          "loss_val: {:.4f}".format(loss_val.item()),
          "acc_val: {:.4f}".format(acc_val.item()),
          "time: {:.4f}s".format(time.time() - t))
    
    if (epoch + 1) % 10 == 0:
        wandb.log({
            "epoch": epoch + 1,
            "loss_train": loss_train.item(),
            "acc_train": acc_train.item(),
            "loss_val": loss_val.item(),
            "acc_val": acc_val.item(),
            "time": time.time() - t
        })


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Save Model
torch.save(model.state_dict(), args.model_path)
print(f"Model saved as {args.model_path}")

wandb.finish()