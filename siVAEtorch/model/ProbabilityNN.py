from typing import List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

from transformers.utils import ModelOutput

from .base import NN

from siVAEtorch.util.configurations import Configuration

class PNNConfiguration(Configuration):

    def __init__(self,
                 input_size: int = 1024,
                 output_dist: str = 'gaus',
                 output_size: int = 2,
                 hidden_layers: Optional[list] = [512, 128],
                 hidden_activation = F.relu,
                 final_activation = None,
                 **kwargs):

        super().__init__(
            input_size = input_size,
            output_dist=output_dist,
            output_size=output_size,
            hidden_layers=hidden_layers,
            hidden_activation=hidden_activation,
            final_activation=final_activation,
            **kwargs
        )


class ProbabilityNN(nn.Module):
    """
    Input
        config: PNNConfiguration
    """
    def __init__(
        self,
        config,
        ):

        super().__init__()

        self.config = config

        # Set prior based on the latent_dim type
        output_dist = config.output_dist
        if output_dist == 'gaus':
            self.prior = Normal(
                torch.zeros(config.output_size),
                torch.ones(config.output_size)
            )
            output_size = config.output_size * 2
        else:
            raise Exception('Input valid output_dist')

        self.nn = NN(
            input_size = config.input_size,
            output_size = output_size,
            hidden_layers = config.hidden_layers,
            hidden_activation = config.hidden_activation,
            final_activation = config.final_activation,
        )


    def forward(self, x, return_hidden_only=False, **kwargs):

        nn_output = self.nn(x, return_hidden=True)
        x = nn_output.x
        h = nn_output.h

        output = ModelOutput(h = h)

        if return_hidden_only:

            pass

        else:

            if self.config.output_dist == 'gaus':

                # Set parameters
                z_mu, z_var = torch.chunk(x,2,dim=-1) # split output into mu,var
                z_var = self.softplus(z_var)

                # Posterior distribution
                z_dist = Normal(z_mu,z_var)

                # Sample
                if self.training:
                    sample = z_dist.rsample()
                else:
                    sample = z_mu

            else:
                raise Exception('Input valid output_dist')

            output.dist=z_dist
            output.mu=z_mu
            output.var=z_var
            output.sample=sample

        return output


    def get_final_weight(self):

        return self.nn.fcs[-1]
