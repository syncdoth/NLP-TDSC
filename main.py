"""
The main file to run experiments.
"""

import argparse

import torch

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
    # training / optimization related
    parser.add_argument('--n_epochs', type=int, default=10, help='number of times to train')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
    parser.add_argument('--decay_steps',
                        type=int,
                        default=0,
                        help='decay lr after x epochs. 0 means to use ReduceLrOnPlateau')
    parser.add_argument('--lr_decay_rate', type=float, default=0.8, help='how much to decay lr')
    parser.add_argument('--patience',
                        type=int,
                        default=3,
                        help='how many epochs to wait until reducing lr; used in ReduceLROnPlateau')
    # extra
    parser.add_argument('--seed', type=int, default=2022, help="the random seed")
    parser.add_argument('--verbose', action='store_true', help='whether to print results a lot')
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default='saved_models/best.pth',
                        help='where to save the model')

    args = parser.parse_args()

    return args


def main():
    args = options()
    set_random_seeds(args.seed)

    MNIST_data = get_MNIST_datasets()
    dataloaders = get_MNIST_dataloaders(MNIST_data, batch_size=args.batch_size, seed=args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FooModel()  # TODO: change FooModel to actual model

    # TODO: add more init & control here
    loss_fn = torch.nn.CrossEntropyLoss()  # TODO: change loss_fn.
    train(model, dataloaders, loss_fn, args, device=device)


if __name__ == '__main__':
    main()
