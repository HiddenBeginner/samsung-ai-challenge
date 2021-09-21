import numpy as np
import gudhi as gd

import torch
from torch_geometric.data import Data

from rdkit import Chem
from descriptastorus.descriptors import rdNormalizedDescriptors  # For generating graph-level features


def row2data(row, encoder_atom, encoder_bond_type, encoder_bond_stereo, encoder_bond_type_stereo):
    smiles = row.SMILES
    y = row.y
    
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    
    # Creating node feature vector
    num_nodes = len(list(m.GetAtoms()))
    x = np.zeros((num_nodes, len(encoder_atom.keys())))
    for i in m.GetAtoms():
        x[i.GetIdx(), encoder_atom[i.GetAtomicNum()]] = 1
    
    x = torch.from_numpy(x).float()

    # Creating edge_index and edge_type
    i = 0
    num_edges = 2 * len(list(m.GetBonds()))
    edge_index = np.zeros((2, num_edges), dtype=np.int64)
    edge_type = np.zeros((num_edges, ), dtype=np.int64)
    for edge in m.GetBonds():
        # Getting bond information
        u = min(edge.GetBeginAtomIdx(), edge.GetEndAtomIdx())
        v = max(edge.GetBeginAtomIdx(), edge.GetEndAtomIdx())
        bond_type = edge.GetBondTypeAsDouble()
        bond_stereo = edge.GetStereo()
        bond_label = encoder_bond_type_stereo[(bond_type, bond_stereo)]

        # Storing information
        edge_index[0, i] = u
        edge_index[1, i] = v
        edge_index[0, i + 1] = v
        edge_index[1, i + 1] = u
        edge_type[i] = bond_label
        edge_type[i + 1] = bond_label
        i += 2
        
    edge_index = torch.from_numpy(edge_index)    
    edge_type = torch.from_numpy(edge_type)

    # Generating graph-level features
    features = generate_graph_level_features(smiles)
    features = torch.tensor([features]).float()

    # Creating y
    y = torch.tensor([y]).float()
    
    # Wrapping all together
    data = Data(
        x=x, 
        edge_index=edge_index, 
        edge_type=edge_type,
        features = features, 
        y=y, 
        uid=row.uid
    )
    
    return data


def generate_graph_level_features(smiles):
    """
    reference
    ----------
    https://github.com/chemprop/chemprop/blob/9c8ff4074bd89b93f43a21adc49b458b0cab9e7f/chemprop/features/features_generators.py#L110
    """
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    features = generator.process(smiles)[1:]
    features = np.nan_to_num(features, 0)

    return features


def calculate_num_cycles(m):
    """
    Calculate the one dimensional betti number, which indicates the number of cycles (holes)
    """
    simplex_tree = gd.SimplexTree()
    for bond in m.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        st.insert([u, v])  
    st.compute_persistence(persistence_dim_max=True)
    
    return st.betti_numbers()[1]


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
