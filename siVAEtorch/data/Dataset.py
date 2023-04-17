from typing import List, Optional, Tuple, Union, Mapping

import os

import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset


class VAEDataset(Dataset):

    def __init__(self, adata, cuda=True):

        self.adata = adata

        self.X = torch.tensor(adata.X).float()
        self.labels = list(adata.obs.Labels.values)

        if cuda:
          self.X.cuda()

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, item):
        adata_sub = self.adata[item]
        inputs = {'X': adata_sub.X,
                  'labels': adata_sub.obs.Labels}
        return inputs
