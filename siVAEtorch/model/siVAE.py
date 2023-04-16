from torch import nn
from transformers.utils import ModelOutput

from .VAE import VAE,VAEConfiguration
from .ProbabilityNN import ProbabilityNN, PNNConfiguration
from siVAEtorch.util.configurations import Configuration


class siVAEConfiguration(Configuration):

    def __init__(self,
                 gamma: float = 0.05,
                 **kwargs):

        super().__init__(
            gamma=gamma,
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

        return feature_config



class siVAE(nn.Module):

    def __init__(
        self,
        config
    ):

        super().__init__()

        self.config = config

        self.cell_vae    = VAE(config)
        self.feature_vae = VAE(config)


    def forward(
        x,
        **kwargs
    ):

        cell_output    = self.cell_vae(x,return_hidden_only=True)
        feature_output = self.feature_vae(x,)

        combined_output = ModelOutput(
            final = final_dist,
            linear = linear_dist,
        )

        output = ModelOutput(
            cell=cell_output,
            feature=feature_output,
            combined=combined_output
        )

        return decoder_outputs


    def calculate_output_dist(
        self,
        cell_output,
        feature_output
    ):
        """
        Calculate the final output distribution for siVAE based on cell/feature
        """

        cell_h = cell_output.decoder.h
        feature_h = feature_output.decoder.mu

        X = torch.matmul(cell_h,feature_h) + self.cell_vae.get_final_weight().bias
        return X


    def calculate_output_dist_linear(
        self,
        cell_output,
        feature_output,
    ):
        """
        Calculate the final output distribution for siVAE based on cell/feature
        as linear multiplication between the cell and feature latent dimension
        """

        cell_latent = cell_output.encoder.mu
        feature_latent = feature_output.encoder.mu

        return torch.matmul(cell_latent, feature_latent)