from typing import List, Optional, Tuple, Union

from torch import nn
import torch.nn.functional as F

from transformers.utils import ModelOutput

from .ProbabilityNN import ProbabilityNN, PNNConfiguration
from siVAEtorch.util.configurations import Configuration


class VAEConfiguration(Configuration):

    def __init__(self,
                 input_size: int = 1024,
                 output_dist: str = 'gaus',
                 output_size: int = 1024,
                 latent_dim : int = 2,
                 hidden_layers: Optional[list] = [512, 128],
                 hidden_layers_decoder: Union[list,None] = None,
                 hidden_activation = F.relu,
                 final_activation = None,
                 **kwargs):

        super().__init__(
            input_size=input_size,
            latent_dim=latent_dim,
            output_dist=output_dist,
            output_size=output_size,
            hidden_layers=hidden_layers,
            hidden_layers_decoder=hidden_layers_decoder,
            hidden_activation=hidden_activation,
            final_activation=final_activation,
            **kwargs)


    def create_encoder(self):

        encoder_config = PNNConfiguration(
            **self.__dict__
        )

        # Latent dim becomes output for encoder
        output_size = encoder_config.latent_dim
        encoder_config.output_size = output_size

        return encoder_config


    def create_decoder(self):

        decoder_config = PNNConfiguration(
            **self.__dict__
        )

        # Latent dim becomes input for decoder
        input_size = decoder_config.latent_dim
        decoder_config.input_size = input_size

        # Reverse order for decoder
        if self.hidden_layers_decoder is None:
            hidden_layers = decoder_config.hidden_layers[::-1]
            decoder_config.hidden_layers = hidden_layers
        else:
            hidden_layers = decoder_config.hidden_layers_decoder
            decoder_config.hidden_layers = hidden_layers

        return decoder_config


class VAE(nn.Module):

    def __init__(self,
                 config):

        super().__init__()

        self.config = config

        self.encoder_config = config.create_encoder()
        self.decoder_config = config.create_decoder()

        self.encoder = ProbabilityNN(self.encoder_config)
        self.decoder = ProbabilityNN(self.decoder_config)

    def forward(self, x, return_all=True, **kwargs):

        encoder_outputs = self.encoder(x)
        decoder_outputs = self.decoder(encoder_outputs.sample,
                                       **kwargs
                                       )

        if return_all:
            output = ModelOutput(decoder=decoder_outputs,
                                 encoder=encoder_outputs)
        else:
            output = ModelOutput(decoder=decoder_outputs)

        return output
