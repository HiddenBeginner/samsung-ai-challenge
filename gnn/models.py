import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import RGCNConv, global_mean_pool


class RGCN(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_channels, hidden_dims, num_node_features=13):
        super(RGCN, self).__init__()
        # Embedding each atom to embedding_dim vector.
        self.embedding = nn.Embedding(num_node_features, embedding_dim)
        
        # GCN
        self.conv_layers = nn.ModuleList()
        for i in range(len(hidden_channels)):
            if i == 0:
                conv = RGCNConv(embedding_dim, hidden_channels[i], num_relations=4)
            else:
                conv = RGCNConv(hidden_channels[i - 1], hidden_channels[i], num_relations=4)
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
        
    def forward(self, x, edge_index, edge_type, batch):
        if self.embedding is not None: 
            x = torch.nonzero(x, as_tuple=True)[1]
            x = self.embedding(x)

        # 1. Obtain node embeddings through gnn
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index, edge_type)
            x = F.normalize(x, 2, 1)
        
        # 2. graph pooling
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        
        # 3. Downstream mlp
        for i, fc in enumerate(self.fc_layers):
            x = F.relu(fc(x))        
        x = self.out(x)
        
        return x


class RGCNSkipConnection(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_channels, hidden_dims, num_node_features=13):
        super(RGCNSkipConnection, self).__init__()
        self.embedding = nn.Embedding(num_node_features, embedding_dim)
        
        # GCN
        self.conv_layers = nn.ModuleList()
        for i in range(len(hidden_channels)):
            if i == 0:
                conv = RGCNConv(embedding_dim, hidden_channels[i], num_relations=6)
            else:
                conv = RGCNConv(hidden_channels[i - 1], hidden_channels[i], num_relations=6)
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

        self.prelu = nn.PReLU()
        
    def forward(self, x, edge_index, edge_type, batch):
        if self.embedding is not None: 
            x = torch.nonzero(x, as_tuple=True)[1]
            x = self.embedding(x)

        # 1. Obtain node embeddings through gnn
        for i, conv in enumerate(self.conv_layers):
            skip = x
            x = conv(x, edge_index, edge_type)
            x = self.prelu(x + skip)
            x = F.normalize(x, 2, 1)
        
        # 2. graph pooling
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        
        # 3. Downstream mlp
        for i, fc in enumerate(self.fc_layers):
            x = F.relu(fc(x))        
        x = self.out(x)
        
        return x
