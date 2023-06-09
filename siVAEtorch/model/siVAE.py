from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.utils import ModelOutput
from torch.distributions import Normal

from siVAEtorch.model.VAE import VAE,VAEConfiguration
from siVAEtorch.model.ProbabilityNN import ProbabilityNN, PNNConfiguration
from siVAEtorch.util.configurations import Configuration


class siVAEConfiguration(Configuration):

    def __init__(self,
                 gamma: float = 0.05,
                 n_cells: int = 500,
                 **kwargs):

        super().__init__(
            gamma=gamma,
            n_cells=n_cells,
            **kwargs)

    def create_cell_config(self):
        """"""

        cell_config = VAEConfiguration(
            **self.__dict__
        )

        return cell_config

    def create_feature_config(self):
        """"""

        feature_config = VAEConfiguration(
            **self.__dict__
        )

        feature_config.input_size = self.n_cells

        feature_config.output_size = feature_config.hidden_layers[::-1][-1]
        feature_config.hidden_layers_decoder = feature_config.hidden_layers[::-1][:-1]

        return feature_config



class siVAE(nn.Module):

    def __init__(
        self,
        config
    ):

        super().__init__()

        self.config = config

        self.cell_vae    = VAE(config.create_cell_config())
        self.feature_vae = VAE(config.create_feature_config())

        self.linear_bias = torch.ones(self.cell_vae.config.output_size).cuda()

        self.X_t = None

    def set_transpose(self, X_t):
        self.X_t = X_t

    def save_feature_embedding(self, feature_ds):

        v_mu = []
        w_mu = []

        with torch.no_grad():
            for inputs in feature_ds:
                outputs = self.feature_vae(inputs['X'])
                v_mu.append(outputs.encoder.mu)
                w_mu.append(outputs.decoder.mu)

        v_mu = torch.stack(v_mu)
        w_mu = torch.stack(w_mu)

        self.feature_embeddings = ModelOutput(v=v_mu,
                                              w=w_mu)


    def forward(
        self,
        X,
        target=None,
        **kwargs
    ):

        cell_output    = self.cell_vae(X)
        feature_output = self.feature_vae(self.X_t)

        final_dist  = self.calculate_output_dist(cell_output, feature_output)
        linear_dist = self.calculate_output_dist_linear(cell_output, feature_output)

        combined_output = ModelOutput(
            final = final_dist,
            linear = linear_dist,
        )

        output = ModelOutput(
            cell=cell_output,
            feature=feature_output,
            combined=combined_output
        )

        return output


    def calculate_output_dist(
        self,
        cell_output,
        feature_output
    ):
        """
        Calculate the final output distribution for siVAE based on cell/feature
        """

        cell_h = cell_output.decoder.h
        # feature_h = feature_output.decoder.mu
        feature_h = feature_output.decoder.mu.transpose(0,1)

        bias = self.cell_vae.decoder.get_final_weight().bias
        bias = torch.chunk(bias,2,dim=-1)[0]
        X = torch.matmul(cell_h,feature_h) + bias

        dist = Normal(X,
                      cell_output.decoder.dist.scale)

        return dist


    def calculate_output_dist_linear(
        self,
        cell_output,
        feature_output
    ):
        """
        Calculate the final output distribution for siVAE based on cell/feature
        as linear multiplication between the cell and feature latent dimension
        """

        cell_latent = cell_output.encoder.mu
        feature_latent = feature_output.encoder.mu.transpose(0,1)

        X = torch.matmul(cell_latent, feature_latent)

        dist = Normal(X,self.linear_bias)

        return dist
