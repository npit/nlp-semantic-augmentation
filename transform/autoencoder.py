"""Module for autoencoder components
"""
import torch
from learning.neural.utils import (make_linear, make_linear_chain,
                                   run_linear_chain)
from serializable import Serializable
from transform.transform import Transform


class SimpleAE(torch.nn.Module):
    """Simple Autoencoder
    """
    def __init__(self, input_dim, target_dim, architecture):
        """Constructor for a simple autoencoder

        Arguments
            input_dim {int} -- The input features dimension
            target_dim {int} -- The reduced features dimension
            architecture {list} -- List of layer dimensions to insert between the input and reduced dimension
        """
        super().__init__()
        self.fcs = make_linear_chain(input_dim, architecture + [target_dim] + list(reversed(architecture)))

    def forward(self, inputs):
        """Forward function for the AE class
        """
        return run_linear_chain(self.fcs, inputs)


class DAE(SimpleAE):
    """Denoising autoencoder"""

    def __init__(self, input_dim, target_dim, noise_coefficient):
        super().__init__()
        self.noise_coefficient = noise_coefficient

    def forward(self, inputs):
        # apply noise in [0, 1] * coeff
        noisy_inputs = (torch.randn(inputs.shape) * self.noise_coefficient).mul(inputs)
        # apply the AE
        return super().forward(noisy_inputs)


class VAE(torch.nn.Module):
    """Variational autoencoder

    Autoencoder instance that regularizes the compressed, latent space
    to ensure a degree of regularity and uniformity in it. Via this, sampling from
    the latent space and decoding can be assumed to be a data generation process.
    Assumes a normal distribution on the latent space vectors.
    """
    def __init__(self, input_dim, target_dim, architecture):
        """Constructor for a VAE

        Arguments
            input_dim {int} -- The input features dimension
            target_dim {int} -- The reduced features dimension
            architecture {list} -- List of layer dimensions to insert between the input and reduced dimension
        """
        super().__init__()
        # make encoding chain
        self.encoder = make_linear_chain(input_dim, architecture[:-1])
        # variables for the probability distribution of the latent space
        self.mu = make_linear(architecture[-1], target_dim)
        self.var = make_linear(architecture[-1], target_dim)

        # make decoding chain
        self.decoder = make_linear_chain(target_dim, list(reversed(architecture))[1:])

    def forward(self, input_data):
        """Forward pass function for the VAE"""
        # encode input
        enc = run_linear_chain(self.encoder, input_data)
        # get pdf parameters
        mu, log_var = self.mu(enc), self.var(enc)
        # sample from the pdf
        # get variance
        var = torch.exp(log_var / 2)
        # get desired distribution by multiplying variance and adding mean to N(0,1)
        sample = torch.randn(var.shape).mul(var).add(mu)
        # decode the sample
        dec = run_linear_chain(self.decoder, sample)
        return dec

def __init__(self, config):
    """PCA constructor"""
    Transform.__init__(self, config)
    self.transformer = None
    self.process_func_train = self.transformer.fit_transform
    self.process_func_test = self.transformer.transform
