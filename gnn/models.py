import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    """
    references
    ----------
    https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing
    """
    def __init__(self, embedding_dim, hidden_channels, hidden_dims, num_node_features=13, dropout=0.5):
        super(GCN, self).__init__()
        self.embedding = nn.Embedding(num_node_features, embedding_dim) if embedding_dim is not None else None
        embedding_dim = num_node_features
        
        # GNN
        self.conv_layers = nn.ModuleList()
        for i in range(len(hidden_channels)):
            if i == 0:
                conv = GCNConv(embedding_dim, hidden_channels[i])
            else:
                conv = GCNConv(hidden_channels[i - 1], hidden_channels[i])
            self.conv_layers.append(conv)

        # MLP
        self.fc_layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i == 0:
                fc = nn.Linear(hidden_channels[-1], hidden_dims[i])
            else:
                fc = nn.Linear(hidden_dims[i - 1], hidden_dims[i])
            self.fc_layers.append(fc)
        self.out = nn.Linear(hidden_dims[-1], 1)

        self.dropout = dropout
        
    def forward(self, x, edge_index, batch):
        if self.embedding is not None: 
            x = torch.nonzero(x, as_tuple=True)[1]
            x = self.embedding(x)

        # 1. Obtain node embeddings through gnn
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            if i != len(self.conv_layers) - 1:
                x = x.relu()
        
        # 2. graph pooling
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        
        # 3. Downstream mlp
        for fc in self.fc_layers:
            # x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(fc(x))
        x = self.out(x)
        
        return x
 