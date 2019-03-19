from torch.utils.data.dataset import Dataset
import datetime
import random
import numpy as np


class DatasetABC(Dataset):
    '''
    Pytroch dataset
    '''
    def __init__(self, db_path, typeIn, size=[64, 64], augmentation_para={'keys': 0}):
        '''
        You own pytorch dataset
        '''
        self.size = size
        pass

    def __getitem__(self, idx):
        '''
        That's how do I format the output
        '''
        sample = {'data': {'name': 'name', 'rgb': np.zeros(self.size)},
                  'target': {'gt': np.ones(self.size)}
                  }
        return sample

    def __len__(self):
        return len(self.df)


class DatasetCombined(Dataset):
    '''
    combine two datasets, return
    '''

    def __init__(self, datasets, ratios=[]):
        self.datasets = datasets
        self.ratios = ratios

    def __getitem__(self, idx):
        np.random.seed(idx + datetime.datetime.now().second + datetime.datetime.now().microsecond)
        dataset_idx = np.random.choice(list(range(len(self.datasets))), p=self.ratios)
        dataset = self.datasets[dataset_idx]
        sample = dataset.__getitem__(idx) if idx<len(dataset) else dataset.__getitem__(random.randint(0, len(dataset)-1)) # equal for many dataset
        return sample

    def __len__(self):
        return max(len(d) for d in self.datasets)
