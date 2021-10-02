import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import RGCNConv, global_add_pool


class NodeEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(NodeEncoder, self).__init__()
        self.encoder = nn.Linear(input_dim, embedding_dim)

    def forward(self, x):
        return self.encoder(x)


class RGCNSkipConnection(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_dims, num_node_features=13, node_embedding_dim=256, dropout=0.3):
        super(RGCNSkipConnection, self).__init__()
        self.node_encoder = NodeEncoder(num_node_features, node_embedding_dim)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        
        # GCN
        self.conv_layers = nn.ModuleList()
        for i in range(len(hidden_channels)):
            if i == 0:
                conv = RGCNConv(node_embedding_dim, hidden_channels[i], num_relations=6, aggr='add')
            else:
                conv = RGCNConv(hidden_channels[i - 1], hidden_channels[i], num_relations=6, aggr='add')
            self.conv_layers.append(conv)

        self.graph_pooling = nn.Sequential(
            nn.Linear(hidden_channels[-1], hidden_channels[-1]),
            nn.ReLU(),
            nn.Linear(hidden_channels[-1], hidden_channels[-1]),
        )

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
        # 1. Encode each atom to a vector representation
        x = self.node_encoder(x)

        # 2. Obtain node embeddings through gnn
        for i, conv in enumerate(self.conv_layers):
            skip = x
            x = conv(x, edge_index, edge_type)
            x = self.prelu(x + skip)
            x = F.normalize(x, 2, 1)
        
        # 3. graph pooling
        x = self.graph_pooling(x)
        x = global_add_pool(x, batch)  # [batch_size, hidden_channels]
        
        # 4. Readout phase
        for i, fc in enumerate(self.fc_layers):
            x = fc(x) 
            x = self.dropout(x)
            x = F.relu(x)

        # 5. Final prediction
        x = F.relu(self.out(x))
        
        return x
