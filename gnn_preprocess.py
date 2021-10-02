import sys
sys.path.append('.')

from gnn.utils import row2data

import os
import argparse
import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Data

import rdkit
import rdkit.Chem as Chem

from collections import defaultdict


def create_encoders(df):
    encoder_atom = defaultdict(lambda : len(encoder_atom))
    encoder_bond_type = defaultdict(lambda : len(encoder_bond_type))
    encoder_bond_stereo = defaultdict(lambda : len(encoder_bond_stereo))
    encoder_bond_type_stereo = defaultdict(lambda : len(encoder_bond_type_stereo))
        
    target = df['SMILES'].values
    total_num = len(target)
    for i, smiles in enumerate(target):
        print(f'Creating the label encoders for atoms, bond_type, and bond_stereo ... [{i + 1} / {total_num}] done !', end='\r')
        m = Chem.MolFromSmiles(smiles)
        m = Chem.AddHs(m)
        
        for atom in m.GetAtoms():
            encoder_atom[atom.GetAtomicNum()]
            
        for bond in m.GetBonds():
            encoder_bond_type[bond.GetBondTypeAsDouble()]
            encoder_bond_stereo[bond.GetStereo()]
            encoder_bond_type_stereo[(bond.GetBondTypeAsDouble(), bond.GetStereo())]
        
    return encoder_atom, encoder_bond_type, encoder_bond_stereo, encoder_bond_type_stereo


def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str, default='./data')
    parser.add_argument('--dir_output', type=str, default='./outputs/gnn')
    args = parser.parse_args()
    
    return args


def main(args):
    dir_data = args.dir_data
    dir_output = args.dir_output

    train = pd.read_csv(f'{dir_data}/train.csv')
    dev = pd.read_csv(f'{dir_data}/dev.csv')
    test = pd.read_csv(f'{dir_data}/test.csv')

    full = pd.concat([train, dev, test], axis=0, ignore_index=True)
    full['y'] = full['S1_energy(eV)'] - full['T1_energy(eV)']
    full['folder'] = full['uid'].apply(lambda x: x.split('_')[0])

    encoder_atom, encoder_bond_type, encoder_bond_stereo, encoder_bond_type_stereo = create_encoders(full)

    if not os.path.exists(dir_output):
        os.makedirs(f'{dir_output}/train')
        os.mkdir(f'{dir_output}/dev')
        os.mkdir(f'{dir_output}/test')

    print('')
    for i, row in full.iterrows():
        print(f'Converting each data into torch.Data ... [{i+1} / {len(full)}] done !', end='\r')
        data = row2data(row, encoder_atom, encoder_bond_type, encoder_bond_stereo, encoder_bond_type_stereo)

        fpath = f'{dir_output}/{row.folder}/{row.uid}.pt'
        torch.save(data, fpath)
        

if __name__ == '__main__':
    args = get_argument()
    main(args)
