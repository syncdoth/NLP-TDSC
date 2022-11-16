# coding: utf-8
# Jiayang Cheng

import torch
from torchvision import datasets, transforms
import numpy as np

from IPython import embed

import scipy.io as sio

def get_MNIST_dataloaders(data_path, data_num=-1, seed=1):
    """ Get MNIST dataloaders, including splitting the original train set into train and validation sets.
    """
    train_data = datasets.MNIST(root=data_path, transform=None, train=True, download=True)

    if data_num < 0:
        data_num = train_data.data.size(0)

    np.random.seed(seed)
    train_data_indices = torch.Tensor(np.random.choice(train_data.data.size(0), data_num, replace=False)).long()

    Datasets = {
        'data': train_data.data[train_data_indices].float(),
        'label': train_data.targets[train_data_indices],
    }

    return Datasets

def get_coil20_dataset(data_path):

    data = sio.loadmat(f'{data_path}/COIL20.mat')
    x, y = data['fea'].reshape((-1, 32, 32)), data['gnd']
    y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]

    Dataset = {
        'data': torch.Tensor(x).float(),
        'label': torch.Tensor(y).long() 
    }

    return Dataset


if __name__ == '__main__':

    dataset = get_coil20_dataset('./data')
    embed()