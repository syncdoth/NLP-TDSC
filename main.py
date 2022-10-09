"""
The main file to run experiments.
"""

import argparse

from util import set_random_seeds
from data import get_MNIST_datasets, get_MNIST_dataloaders
from train import train
from model import FooModel


def options():
    """
    add command line options here
    """
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=2022, help="the random seed")
    args = parser.parse_args()

    return args


def main():
    args = options()
    set_random_seeds(args.seed)

    MNIST_data = get_MNIST_datasets()
    dataloaders = get_MNIST_dataloaders(MNIST_data,
                                        batch_size=args.batch_size,
                                        seed=args.seed)

    model = FooModel()  # TODO: change FooModel to actual model

    # TODO: add more init & control here
    train(model, dataloaders, args=args)


if __name__ == '__main__':
    main()
