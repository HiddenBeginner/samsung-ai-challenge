import numpy as np

import torch
from torch_geometric.data import Data

import rdkit.Chem as Chem


BOND_TYPE_ENCODER = {1.0: 0, 2.0: 1, 1.5: 2, 3.0: 3}


def row2data(row, encoder):
    smiles = row.SMILES
    y = row.y
    
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    
    # Creating node feature vector
    num_nodes = len(list(m.GetAtoms()))
    x = np.zeros((num_nodes, len(encoder.keys())))
    for i in m.GetAtoms():
        x[i.GetIdx(), encoder[i.GetAtomicNum()]] = 1
    
    x = torch.from_numpy(x).float()

    # Creating edge_index
    i = 0
    num_edges = 2 * len(list(m.GetBonds()))
    edge_index = np.zeros((2, num_edges), dtype=np.int64)
    edge_type = np.zeros((num_edges, ), dtype=np.int64)
    for edge in m.GetBonds():
        u = min(edge.GetBeginAtomIdx(), edge.GetEndAtomIdx())
        v = max(edge.GetBeginAtomIdx(), edge.GetEndAtomIdx())
        edge_index[0, i] = u
        edge_index[1, i] = v
        edge_index[0, i + 1] = v
        edge_index[1, i + 1] = u
        edge_type[i] = BOND_TYPE_ENCODER[edge.GetBondTypeAsDouble()]
        edge_type[i + 1] = BOND_TYPE_ENCODER[edge.GetBondTypeAsDouble()]
        i += 2
        
    edge_index = torch.from_numpy(edge_index)    
    edge_type = torch.from_numpy(edge_type)
    # Creating y
    y = torch.tensor([y]).float()
    
    # Wrapping all together
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y, uid=row.uid)
    
    return data


class AverageMeter:
    '''
    Compute and store the average and current value
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
