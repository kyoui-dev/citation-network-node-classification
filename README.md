# Citation Network Node Classification

## Setup

### Prerequisites
- Python 3.11

### Installation

#### Install PyTorch
```
pip install "torch==2.3.0" --index-url https://download.pytorch.org/whl/cpu
```

#### Install PyG dependency extensions
```
pip install \
  torch-scatter \
  torch-sparse \
  torch-cluster \
  torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
```

#### Install PyTorch Geometric
```
pip install torch-geometric
```

#### Install remaining requirements
```
pip install -r requirements.txt
```

## Usage

### GCN

#### Train model
```
python -m gcn.train
```

#### Inference
```
python -m gcn.inference
```

### GAT

#### Train model
```
python -m gat.train
```

#### Inference
```
python -m gat.inference
```