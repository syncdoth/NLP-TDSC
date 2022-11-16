"""
The main file to run experiments.
"""

import argparse

import torch
from util import set_random_seeds
from data import get_MNIST_dataloaders, get_coil20_dataset
from train import train
from model import create_ae, SCNet, self_exp, Kfactor
from IPython import embed
import os

def options():
    """
    add command line options here
    """
    parser = argparse.ArgumentParser()
    # add arguments

    # model settings
    parser.add_argument('--data_type', type=str, default='mnist', choices=['mnist', 'coil20'])
    parser.add_argument('--data_num', type=int, default=-1)

    # training / optimization related
    parser.add_argument('--n_epochs', type=int, default=400, help='number of times to train')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
    parser.add_argument('--gamma0', type=float, default=1.0)
    parser.add_argument('--gamma1', type=float, default=1.0)
    parser.add_argument('--gamma2', type=float, default=1.0)
    
    # cluster setting
    parser.add_argument('--cluster_num', type=int, default=10)
    parser.add_argument('--sc_type', type=str, default='spectral', choices=['kfactor', 'spectral'])
    parser.add_argument('--loss_type', type=str, default='normal', choices=['normal', 'triplet'])
    parser.add_argument('--alpha', type=float, default=0.04)
    parser.add_argument('--dim_subspace', type=int, default=32)
    parser.add_argument('--rho', type=int, default=8)

    parser.add_argument('--initialize', action='store_true')
    parser.add_argument('--pretrain_weights', type=str, default='./pretrain_weights')
    
    # extra
    parser.add_argument('--seed', type=int, default=2022, help="the random seed")
    parser.add_argument('--verbose', action='store_true', help='whether to print results a lot')
    parser.add_argument('--output_dir', type=str, default=None, help='where to save the model')
    parser.add_argument('--screen_epoch', type=int, default=10)
    parser.add_argument('--save_epoch', type=int, default=10)
    args = parser.parse_args()
    return args

def main():
    args = options()
    set_random_seeds(args.seed)

    if args.output_dir is None:
        args.output_dir = f"./_output_{args.sc_type}_{args.loss_type}_{args.data_type}"
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.pretrain_weights, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.data_type == 'mnist':
        dataloaders = get_MNIST_dataloaders('./data', data_num=args.data_num, seed=args.seed)
        if args.data_num < 0:
            args.data_num = dataloaders['data'].size(0)
    elif args.data_type == 'coil20': 
        dataloaders = get_coil20_dataset('./data')
        args.data_num = dataloaders['data'].size(0) 
    
    if args.batch_size < 0:
        args.batch_size = args.data_num

    encoder, decoder = create_ae(kernels=[5, 3, 3], channels=[1, 20, 10, 5])
    if args.sc_type == 'spectral':
        cluster_m = self_exp(size=args.data_num, gamma0=args.gamma0, gamma1=args.gamma1)
    elif args.sc_type == 'kfactor':
        cluster_m = Kfactor(cluster_num=args.cluster_num, factor_dim=(dataloaders['data'].size(-1)//8)**2 *5, factor_num_per_cluster=args.dim_subspace, gamma1=args.gamma1)
    model = SCNet(encoder, decoder, cluster_m, pretrain=args.initialize)
    model = model.to(device)
    print(args)
    train(model, dataloaders, args, device=device)

if __name__ == '__main__':
    main()
