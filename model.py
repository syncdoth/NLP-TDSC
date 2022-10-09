"""
Contains the implementation of models
"""

from torch import nn


class FooModel(nn.Module):
    """
    This is a "foo" model, i.e., just an example of pytorch model.
    """

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)
