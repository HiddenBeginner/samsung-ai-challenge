import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, RGCNConv, BatchNorm, Sequential
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_channels, hidden_dims, num_node_features=13):
        super(GCN, self).__init__()
        if embedding_dim is not None:
            self.embedding = nn.Embedding(num_node_features, embedding_dim)
        else:
            self.embedding = None
            embedding_dim = num_node_features
        
        # GNN
        self.conv_layers = nn.ModuleList()
        for i in range(len(hidden_channels)):
            if i == 0:
                conv = Sequential('x, edge_index', [
                    (GCNConv(embedding_dim, hidden_channels[i]), 'x, edge_index -> x'),
                    BatchNorm(hidden_channels[i]),
                    nn.PReLU()
                ])
            else:
                conv = Sequential('x, edge_index', [
                    (GCNConv(hidden_channels[i - 1], hidden_channels[i]), 'x, edge_index -> x'),
                    BatchNorm(hidden_channels[i]),
                    nn.PReLU()
                ])
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
        
    def forward(self, x, edge_index, batch):
        if self.embedding is not None: 
            x = torch.nonzero(x, as_tuple=True)[1]
            x = self.embedding(x)

        # 1. Obtain node embeddings through gnn
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            x = F.normalize(x, 2, 1)
        
        # 2. graph pooling
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        
        # 3. Downstream mlp
        for i, fc in enumerate(self.fc_layers):
            x = F.relu(fc(x))        
        x = self.out(x)
        
        return x
 

class SkipConnectionGCN(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_channels, hidden_dims, num_node_features=13):
        super(SkipConnectionGCN, self).__init__()
        if embedding_dim is not None:
            self.embedding = nn.Embedding(num_node_features, embedding_dim)
        else:
            self.embedding = None
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
        
    def forward(self, x, edge_index, batch):
        if self.embedding is not None: 
            x = torch.nonzero(x, as_tuple=True)[1]
            x = self.embedding(x)

        # 1. Obtain node embeddings through gnn
        for i, conv in enumerate(self.conv_layers):
            skip_connection = x
            x = conv(x, edge_index) + skip_connection
            if i != len(self.conv_layers) - 1:
                x = F.relu(x)
        
        # 2. graph pooling
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        
        # 3. Downstream mlp
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        x = self.out(x)
        
        return x


class RGCN(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_channels, hidden_dims, num_node_features=13):
        super(RGCN, self).__init__()
        if embedding_dim is not None:
            self.embedding = nn.Embedding(num_node_features, embedding_dim)
        else:
            self.embedding = None
            embedding_dim = num_node_features
        
        # GNN
        self.conv_layers = nn.ModuleList()
        for i in range(len(hidden_channels)):
            if i == 0:
                conv = Sequential('x, edge_index, edge_type', [
                    (RGCNConv(embedding_dim, hidden_channels[i], num_relations=4), 'x, edge_index, edge_type -> x'),
                    BatchNorm(hidden_channels[i]),
                    nn.PReLU()
                ])
            else:
                conv = Sequential('x, edge_index, edge_type', [
                    (RGCNConv(hidden_channels[i - 1], hidden_channels[i], num_relations=4), 'x, edge_index, edge_type -> x'),
                    BatchNorm(hidden_channels[i]),
                    nn.PReLU()
                ])
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
        if embedding_dim is not None:
            self.embedding = nn.Embedding(num_node_features, embedding_dim)
        else:
            self.embedding = None
            embedding_dim = num_node_features
        
        # GNN
        self.conv_layers = nn.ModuleList()
        for i in range(len(hidden_channels)):
            if i == 0:
                conv = Sequential('x, edge_index, edge_type', [
                    (RGCNConv(embedding_dim, hidden_channels[i], num_relations=4), 'x, edge_index, edge_type -> x'),
                    BatchNorm(hidden_channels[i]),
                ])
            else:
                conv = Sequential('x, edge_index, edge_type', [
                    (RGCNConv(hidden_channels[i - 1], hidden_channels[i], num_relations=4), 'x, edge_index, edge_type -> x'),
                    BatchNorm(hidden_channels[i]),
                ])
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
