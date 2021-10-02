import sys
sys.path.append('.')

from config import TrainConfig
from gnn.models import RGCNSkipConnection
from gnn.solver import RGCNSolver
from gnn.datasets import GNNDataset

import torch
from torch_geometric.data import DataLoader
import pandas as pd


def main():
    config = TrainConfig
    torch.manual_seed(config.seed)

    # Loading and spliting datasets
    dataset = GNNDataset(f'{config.dir_data}/train')
    train_dataset = dataset[:27000]
    valid_dataset = dataset[27000:]
    dev_dataset = GNNDataset(f'{config.dir_data}/dev')
    test_dataset = GNNDataset(f'{config.dir_data}/test')
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Defining model and solver
    model = RGCNSkipConnection(
        config.hidden_channels,
        config.hidden_dims,
        config.num_node_features,
        config.node_embedding_dim,
        config.dropout
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    solver = RGCNSolver(model, config.lr, config.n_epochs, device)

    # Training model
    solver.fit(train_loader, valid_loader, dev_loader)

    # Predicting
    sub = pd.read_csv('./data/sample_submission.csv')

    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data.to(device)
            pred = model(data.x, data.edge_index, data.edge_type, data.batch).detach().cpu().item()
            sub.loc[sub['uid'] == data.uid[0], 'ST1_GAP(eV)'] = pred
    
    sub.to_csv('single_model_submission.csv', index=False)


if __name__ == '__main__':
    main()