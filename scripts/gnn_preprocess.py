import sys
sys.path.append('.')

from gnn.utils import row2data

import os
import argparse
import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Data

import rdkit.Chem as Chem


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

    if not os.path.exists(dir_output):
        os.makedirs(f'{dir_output}/train')
        os.mkdir(f'{dir_output}/dev')
        os.mkdir(f'{dir_output}/test')

    for i, row in full.iterrows():
        print(f'\r[{i+1} / {len(full)}] Done', end='')
        data = row2data(row)

        fpath = f'{dir_output}/{row.folder}/{row.uid}.pt'
        torch.save(data, fpath)
        

if __name__ == '__main__':
    args = get_argument()
    main(args)
