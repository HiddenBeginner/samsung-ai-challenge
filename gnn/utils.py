import numpy as np

import torch
from torch_geometric.data import Data

import rdkit.Chem as Chem


def row2data(row, max_atoms=100):
    smiles = row.SMILES
    y = row.y
    
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    
    # Creating node feature vector
    num_nodes = len(list(m.GetAtoms()))
    x = np.zeros((num_nodes, max_atoms))
    for i in m.GetAtoms():
        x[i.GetIdx(), i.GetAtomicNum()] = 1
    
    x = torch.from_numpy(x).float()

    # Creating edge_index
    i = 0
    num_edges = 2 * len(list(m.GetBonds()))
    edge_index = np.zeros((2, num_edges), dtype=np.int64)
    for edge in m.GetBonds():
        u = min(edge.GetBeginAtomIdx(), edge.GetEndAtomIdx())
        v = max(edge.GetBeginAtomIdx(), edge.GetEndAtomIdx())
        edge_index[0, i] = u
        edge_index[1, i] = v
        edge_index[0, i + 1] = v
        edge_index[1, i + 1] = u
        i += 2
        
    edge_index = torch.from_numpy(edge_index)    
    
    # Creating y
    y = torch.tensor([y]).float()
    
    # Wrapping all together
    data = Data(x=x, edge_index=edge_index, y=y)
    
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
