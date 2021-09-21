import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import RGCNConv, global_mean_pool


class NodeEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(NodeEncoder, self).__init__()
        self.encoder = nn.Linear(input_dim, embedding_dim)

    def forward(self, x):
        return self.encoder(x)


class RGCNSkipConnection(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_channels, hidden_dims, num_node_features=13, num_graph_features=200):
        super(RGCNSkipConnection, self).__init__()
        self.node_encoder = NodeEncoder(num_node_features, embedding_dim)
        
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
                fc = nn.Linear(hidden_channels[-1] + num_graph_features, hidden_dims[i])
            else:
                fc = nn.Linear(hidden_dims[i - 1], hidden_dims[i])
            self.fc_layers.append(fc)
        self.out = nn.Linear(hidden_dims[-1], 1)

        self.prelu = nn.PReLU()
        
    def forward(self, x, edge_index, edge_type, features, batch):
        x = self.node_encoder(x)

        # 1. Obtain node embeddings through gnn
        for i, conv in enumerate(self.conv_layers):
            skip = x
            x = conv(x, edge_index, edge_type)
            x = self.prelu(x + skip)
            x = F.normalize(x, 2, 1)
        
        # 2. graph pooling
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = torch.cat((x, features), dim=1)  # [batch_size, hidden_channels + num_graph_features]
        
        # 3. Downstream mlp
        for i, fc in enumerate(self.fc_layers):
            x = F.relu(fc(x))        
        x = self.out(x)
        
        return x
