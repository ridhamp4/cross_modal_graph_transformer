import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj, dense_to_sparse

class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, edge_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.edge_dim = edge_dim
        
        # Node transformations
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
        # Edge transformation
        self.W_e = nn.Linear(edge_dim, num_heads)
        
        # Batch norms
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        # Transform nodes to Q,K,V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Transform edges to attention weights
        E = self.W_e(edge_attr).view(-1, self.num_heads)
        
        # Compute attention scores with edge features
        attn_scores = (Q[edge_index[0]] * K[edge_index[1]]).sum(dim=1)
        attn_scores = attn_scores.view(-1, self.num_heads) + E
        attn_scores = F.leaky_relu(attn_scores)
        
        # Create attention matrix
        adj = to_dense_adj(edge_index, edge_attr=attn_scores).squeeze(0)
        attn_weights = F.softmax(adj, dim=1)
        
        # Apply attention
        x_new = torch.matmul(attn_weights, V)
        
        # Residual + BN
        x = self.bn1(x + x_new)
        
        # FFN
        x_new = self.ffn(x)
        x = self.bn2(x + x_new)
        
        return x

class GraphTransformer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=80, num_layers=4, num_heads=8):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads, edge_dim)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x = self.node_embed(data.x)
        edge_attr = self.edge_embed(data.edge_attr)
        
        for layer in self.layers:
            x = layer(x, data.edge_index, edge_attr)
        
        # Readout
        x = x.mean(dim=0)
        return self.classifier(x)
