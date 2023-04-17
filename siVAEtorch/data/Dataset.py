from typing import List, Optional, Tuple, Union, Mapping

import os

import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset


class VAEDataset(Dataset):

    def __init__(self, adata, cuda=True):

        self.adata = adata

        self.X        = torch.tensor(adata.X).float()
        self.labels   = list(adata.obs.Labels.values)
        self.label_id = np.arange(len(adata.X))

        if cuda:
          self.X.cuda()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        inputs = {'X': self.X[item],
                  'labels': self.labels[item],
                  'label_id': self.label_id[item]}
        return inputs
