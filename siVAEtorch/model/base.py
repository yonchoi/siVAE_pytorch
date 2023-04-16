from typing import List, Optional, Tuple, Union

from transformers.utils import ModelOutput

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchshape import tensorshape

import numpy as np

from tqdm import tqdm

from transformers.utils import ModelOutput

## Set functions for initializing weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class NN(nn.Module):
    """
    Base module for convolutional neural network (CNN) connected to a fully connected network (FCN)
    If n_conv_layers is set to 0 by default and acts as FCN
    """

    def __init__(self,
                 input_shape: Union[list,int],
                 output_size: Optional[int] = None,
                 n_conv_layers: int = 0,
                 hidden_layers: list = [],
                 hidden_activation = F.relu,
                 final_activation = None,
                 kernel_size: int = 10,
                 out_channels: int = 128,
                 stride: int = 1,
                 mp_kernel_size: Optional[int] = 2,
                 mp_stride: Optional[int] = 1,
                 ):

        super().__init__()

        self.output_size = output_size
        # self.loss_fct = loss_fct
        self.hidden_activation = hidden_activation

        # Convolutional nets
        self.pool = nn.MaxPool1d(
            kernel_size=mp_kernel_size,
            stride=mp_stride
            )

        self.convs = nn.ModuleList()

        outshape = [1] + list(np.array(1).reshape(-1))

        if out_channels is None:
            # Set out_channels to in_channels
            out_channels = outshape[1]

        print('CNN')
        for i in range(n_conv_layers):
            print(i,outshape)
            conv = nn.Conv1d(
                in_channels=outshape[1],
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                )
            self.convs.append(conv)
            outshape = tensorshape(conv, outshape)
            print(i,outshape)
            outshape = tensorshape(self.pool, outshape)
            print(i,outshape)

        current_size = np.prod(outshape)

        print('final_size',current_size)

        # Fully connected layers
        if self.output_size is not None:

            self.fcs = nn.ModuleList()

            for h_dim in hidden_layers:
                self.fcs.append(nn.Linear(current_size,h_dim))
                current_size = h_dim
            self.fcs.append(nn.Linear(current_size,output_size))

            self.final_activation = final_activation

            current_size = output_size

        self.final_size = current_size


    def forward(self, x, return_hidden=False, **kwargs):

        for conv in self.convs:
            x = self.hidden_activation(conv(x))
            x = self.pool(x)

        x = torch.flatten(x, 1) # flatten all dimensions except batch

        output = ModelOutput()

        if self.output_size is not None:

            for layer in self.fcs[:-1]:
                x = self.hidden_activation(layer(x))

            x = self.fcs[-1](x)

            if return_hidden:
                output.h = x

            if self.final_activation:
                x = self.final_activation(x)

        output.x = x

        return output
