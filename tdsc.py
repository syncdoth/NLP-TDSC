"""
Contains the implementation of TDSC
Reference: https://github.com/XifengGuo/DSC-Net/blob/master/main.py
"""

import argparse
import math
import os
import warnings
import tqdm

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from post_clustering import acc, nmi, spectral_clustering
from data import get_triplet_data
from util import set_random_seeds


class Conv2dSamePad(nn.Module):
    """
    Implement Tensorflow's 'SAME' padding mode in Conv2d.
    """

    def __init__(self, kernel_size, stride):
        super(Conv2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        out_height = math.ceil(float(in_height) / float(self.stride[0]))
        out_width = math.ceil(float(in_width) / float(self.stride[1]))
        pad_along_height = ((out_height - 1) * self.stride[0] + self.kernel_size[0] - in_height)
        pad_along_width = ((out_width - 1) * self.stride[1] + self.kernel_size[1] - in_width)
        pad_top = math.floor(pad_along_height / 2)
        pad_left = math.floor(pad_along_width / 2)
        pad_bottom = pad_along_height - pad_top
        pad_right = pad_along_width - pad_left
        return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], 'constant', 0)


class ConvTranspose2dSamePad(nn.Module):
    """
    This module implements the "SAME" padding mode for ConvTranspose2d as in Tensorflow.
    """

    def __init__(self, kernel_size, stride):
        super(ConvTranspose2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        pad_height = self.kernel_size[0] - self.stride[0]
        pad_width = self.kernel_size[1] - self.stride[1]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return x[:, :, pad_top:in_height - pad_bottom, pad_left: in_width - pad_right]


class ConvAE(nn.Module):
    def __init__(self, channels, kernels):
        """
        :param channels: a list containing all channels including the input image channel (1 for gray, 3 for RGB)
        :param kernels:  a list containing all kernel sizes, it should satisfy: len(kernels) = len(channels) - 1.
        """
        super(ConvAE, self).__init__()
        assert isinstance(channels, list) and isinstance(kernels, list)
        self.encoder = nn.Sequential()
        for i in range(1, len(channels)):
            #  Each layer will divide the size of feature map by 2
            self.encoder.add_module('pad%d' % i, Conv2dSamePad(kernels[i - 1], 2))
            self.encoder.add_module('conv%d' % i,
                                    nn.Conv2d(channels[i - 1], channels[i], kernel_size=kernels[i - 1], stride=2))
            self.encoder.add_module('relu%d' % i, nn.ReLU(True))

        self.decoder = nn.Sequential()
        channels = list(reversed(channels))
        kernels = list(reversed(kernels))
        for i in range(len(channels) - 1):
            # Each layer will double the size of feature map
            self.decoder.add_module('deconv%d' % (i + 1),
                                    nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernels[i], stride=2))
            self.decoder.add_module('padd%d' % i, ConvTranspose2dSamePad(kernels[i], 2))
            self.decoder.add_module('relud%d' % i, nn.ReLU(True))

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y


class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)

    def forward(self, x):  # shape=[n, d]
        # self.Coefficient.fill_diagonal_(0)    # Error here, due to inplace replacement
        C = self.Coefficient - torch.diag(self.Coefficient)     # to ensure that diag(C) = 0
        y = torch.matmul(C, x)
        return y


class TDSCNet(nn.Module):
    def __init__(self, channels, kernels, num_sample):
        super(TDSCNet, self).__init__()
        self.n = num_sample     # we can use batch_size instead
        self.ae = ConvAE(channels, kernels)
        self.self_expression = SelfExpression(self.n)

    def forward(self, x):   # 3 key:value `anchor, pos, neg', each has shape=[n, c, w, h]
        x_recon, z, z_recon = {}, {}, {}
        for key in ['anchor', 'pos', 'neg']:
            z[key] = self.ae.encoder(x[key])
            x_recon[key] = self.ae.decoder(z[key])   # shape=[n, c, w, h]

            # self expression layer, reshape to vectors, multiply Coefficient, then reshape back
            z[key] = z[key].view(self.n, -1)       # shape=[n, d]
            z_recon[key] = self.self_expression(z[key])   # shape=[n, d]

        return x_recon, z, z_recon

    def loss_fn(self, x, x_recon, z, z_recon, weights):
        loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        loss_triplet = F.triplet_margin_loss(z['anchor'], z['pos'], z['neg'])
        loss_ae, loss_selfExp = 0, 0
        n = x['anchor'].shape[0]

        for key in ['anchor', 'pos', 'neg']:
            # reconstruction loss
            loss_ae += F.mse_loss(x_recon[key], x[key], reduction='sum')
            # (triplet) self-expression loss
            loss_selfExp += F.mse_loss(z_recon[key], z[key], reduction='sum')

        # follow eq 9 -> 12 in the paper. multiply each loss by 2.
        loss = 1/n*loss_ae + weights['c']*loss_coef + weights['se']*loss_selfExp + weights['tri']*loss_triplet
        return loss


def tdscnet_train(model, x, y, unsup_label_init_source='random',
                epochs=10, epochs_update=1, lr=1e-3, weights={'c':1.0,'se':150,'tri':1.0},
                alpha=0.04, dim_subspace=12, ro=8,
                show=10, device='cuda'):

    num_sample = x.shape[0]
    optimizer = optim.Adam(model.parameters(), lr=lr)   # 1e-3 is exactly default lr of Adam
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    x = x.to(device)
    if isinstance(y, torch.Tensor):
        y = y.to('cpu').numpy()
    K = len(np.unique(y))
    if unsup_label_init_source == 'random':
        y_unsup = torch.empty(num_sample, dtype=torch.int8).random_(K)
    else:
        y_unsup = y

    for epoch in tqdm.trange(epochs, desc='Training', position=0):
        D = get_triplet_data(num_sample, y_unsup)
        x_triple = {k:x[D[k]] for k in D.keys()}
        x_recon, z, z_recon = model(x_triple)
        loss = model.loss_fn(x_triple, x_recon, z, z_recon, weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % epochs_update == 0:
            # print('Update unsup label')
            C = model.self_expression.Coefficient.detach().to('cpu').numpy()
            y_unsup = spectral_clustering(C, K, dim_subspace, alpha, ro)
            y_unsup = torch.tensor(y_unsup, dtype=torch.int8)

        if (epoch+1) % show == 0 or epoch == epochs - 1:    # this is evaluation step, on the whole dataset!
            print('Epoch %02d: loss=%.4f, acc=%.4f, nmi=%.4f' %
                  (epoch, loss.item() / y_unsup.shape[0], acc(y, y_unsup.numpy()), nmi(y, y_unsup.numpy())))

    return y_unsup   # for the next unsupervised training pass


def tdscnet_experiments():
    parser = argparse.ArgumentParser(description='TDSCNet')
    parser.add_argument('--db', default='coil20',
                        choices=['coil20', 'coil100', 'orl', 'reuters10k', 'stl'])
    parser.add_argument('--show-freq', default=10, type=int)
    parser.add_argument('--ae-weights', default=None)
    parser.add_argument('--save-dir', default='saved_models/tdsc_cv')
    parser.add_argument('--num_unsup_clusters', type=int, default=2)
    parser.add_argument('--seed', default=100, type=int)
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    set_random_seeds(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    db = args.db

    # TODO: refer to table 2 and 3 in the TDSC paper to set these hyper-parameters
    if db == 'coil20':
        # load data
        data = sio.loadmat('data/COIL20.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]

        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 15]
        kernels = [3]
        epochs = 300        # T_max
        epochs_update = 2   # T_update
        weights={'c':100,'se':0.2,'tri':0.01}
        # weights = {gamma_0, gamma_1, gamma_2}

        # post clustering parameters
        alpha = 0.04  # threshold of C
        dim_subspace = 12  # dimension of each subspace
        ro = 8  #
        warnings.warn("You can uncomment line#64 in post_clustering.py to get better result for this dataset!")

    elif db == 'coil100':
        # load data
        data = sio.loadmat('data/COIL100.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]

        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 50]
        kernels = [5]
        epochs = 120
        epochs_update = 2
        weights={'c':1.0,'se':15,'tri':1.0}

        # post clustering parameters
        alpha = 0.04  # threshold of C
        dim_subspace = 12  # dimension of each subspace
        ro = 8  #

    elif db == 'orl':
        # load data
        data = sio.loadmat('data/ORL_32x32.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 3, 3, 5]
        kernels = [3, 3, 3]
        epochs = 700
        epochs_update = 2
        weights={'c':2.0,'se':0.2,'tri':1.0}

        # post clustering parameters
        alpha = 0.2  # threshold of C
        dim_subspace = 3  # dimension of each subspace
        ro = 1  #

    print('Initialize TDSC model ...')
    tdscnet = TDSCNet(num_sample=num_sample, channels=channels, kernels=kernels).to(device)

    # load the pretrained weights which are provided by the original author in
    # https://github.com/panji1990/Deep-subspace-clustering-networks
    # TDSC paper, page 5, Training Strategy of Network, it said two stages training -> need to load pretrained model
    ae_state_dict = torch.load('pretrained_weights_original/%s.pkl' % db)
    tdscnet.ae.load_state_dict(ae_state_dict)
    print("Pretrained ae weights are loaded successfully.")

    print('Start training ...')
    tdscnet_train(tdscnet, x, y, unsup_label_init_source='random', # 'random' initialization or 'ground_truth'
                epochs=epochs, epochs_update=epochs_update, weights=weights,
                alpha=alpha, dim_subspace=dim_subspace, ro=ro,
                show=args.show_freq, device=device)
    torch.save(tdscnet.state_dict(), args.save_dir + '/%s-model.pt' % args.db)



if __name__ == '__main__':
    with torch.autograd.set_detect_anomaly(True):
        tdscnet_experiments()