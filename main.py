"""
The main file to run experiments.
"""

import argparse

import torch

from util import set_random_seeds
from data import get_MNIST_datasets, get_MNIST_dataloaders
from train import train
from model import FooModel, create_MLP
from IPython import embed

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
    parser.add_argument('--subspace_loss_lambda', type=float, default=0.01)
    parser.add_argument('--lr_decay_rate', type=float, default=0.8, help='how much to decay lr')
    parser.add_argument('--num_samples', type=int, default=1000)
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder = create_MLP(28*28, 16, 64, 1)
    decoder = create_MLP(16, 28*28, 64, 1)
    model = FooModel(encoder, decoder, args.num_samples) 
    loss_fn = lambda x, x_hat: (x-x_hat).square().sum(dim=-1).mean() # reconstruction loss

    train(model, MNIST_data, loss_fn, args, device=device)


if __name__ == '__main__':
    main()
