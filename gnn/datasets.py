from glob import glob

import torch
from torch_geometric.data import InMemoryDataset


class GNNDataset(InMemoryDataset):
    def __init__(self, root):
        super(GNNDataset, self).__init__(root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        pass
    
    def process(self):
        data_list = glob(f'{self.root}/*.pt')
        data_list = list(map(torch.load, data_list))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
