"""
implementation of the training loop.
"""
from torch import nn


def train(model: nn.Module, dataloaders, args=None):
    raise NotImplementedError