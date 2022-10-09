"""
This file contains utility functions.
"""

import os
import random
from typing import Union, List, Tuple

import numpy as np
import torch
from torch import nn


def set_random_seeds(seed):
    """
    set the random seed of all related libraries
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def freeze_net(module: nn.Module, keys: Union[List, Tuple] = None):
    """
    freeze (don't allow training) the weights of a pytorch nn.Module
    """
    for k, p in module.named_parameters():
        if keys is None or k in keys:
            p.requires_grad = False


def unfreeze_net(module: nn.Module):
    """
    un-freeze (allow training) the weights of a pytorch nn.Module
    """
    for p in module.parameters():
        p.requires_grad = True


if __name__ == '__main__':
    test_layer = nn.Linear(1, 1)
    freeze_net(test_layer, keys=['weight'])
    print([
        f'{k} requires_grad: {p.requires_grad}' for k, p in test_layer.named_parameters()
    ])
    unfreeze_net(test_layer)
    print([
        f'{k} requires_grad: {p.requires_grad}' for k, p in test_layer.named_parameters()
    ])
