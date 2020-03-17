"""Module of low-level neural network building blocks

The module is dependent on the DNN backend framework. Pytorch is currently being used."""
import torch
from torch import nn
from torch.nn import functional as F


def make_linear(input_dim, output_dim):
    """Make a linear fully-connected layer
    """
    return nn.Linear(input_dim, output_dim)


def make_linear_chain(input_dim, dim_list):
    """Make a chain of connected linear layers
    Arguments:
        input_dim {int} -- Input dimension
        dim_list {int} -- Dimension of subsequent layers
    Returns:

    """
    layers = []
    current = input_dim
    # make the chain
    for dim in dim_list:
        layers.append(make_linear(current, dim))
        current = dim
    # return as a module list
    return nn.ModuleList(layers)


def run_linear_chain(layers_chain, input_data, activation_func=None):
    """Run a chain of connected linear layers
    Arguments:
        input_data {torch.Tensor} -- Input data
        layers_chain {nn.ModuleList} -- List of layers to execute in sequence
        activation_func {function} -- Activation func to apply after each layer
    Returns:

    """
    # ReLU activations by default
    if activation_func is None:
        activation_func = F.relu
    current_data = input_data
    # make the chain
    for layer in layers_chain:
        current_data = layer(current_data)
        current_data = activation_func(current_data)
    return current_data


def reconstruction_loss(estimate, gt):
    """Return the reconstruction loss of the estimate versus the ground truth"""
    return F.binary_cross_entropy(estimate, gt, reduction='sum')


def kl_divergence(mu, log_variance):
    """Return the reconstruction loss of the estimate versus the ground truth"""
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + log_variance - mu.pow(2) - log_variance.exp())
    return KLD
