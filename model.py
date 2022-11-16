"""
Contains the implementation of models
"""

from symbol import factor
import torch
from torch import nn
import torch.nn.functional as F



class Kfactor(nn.Module):
    """
    SelfExpression module, implement K-FACTORIZATION SUBSPACE CLUSTERING introduced in https://dl.acm.org/doi/pdf/10.1145/3447548.3467267
    """

    def __init__(self, cluster_num, factor_dim=128, factor_num_per_cluster=64, gamma1=1.0, DTD_freq = 10):
        super().__init__()
        assert factor_dim > factor_num_per_cluster

        self.cluster_num = cluster_num
        self.factor_dim = factor_dim
        self.factor_num_per_cluster = factor_num_per_cluster
        self.gamma1 = gamma1

        # self.D = nn.Parameter(torch.randn((cluster_num, factor_dim, factor_num_per_cluster)))

        self.register_buffer('D', torch.ones(cluster_num, factor_dim, factor_num_per_cluster))

    def forward(self, x):
        """
        Compute C given D, this process is not involed in backward propagation
        """
        
        with torch.no_grad():
            Cs = torch.einsum('nij,bi->nbj', self.D, x)
            x_hat = torch.einsum('nij,nbj->nbi', self.D, Cs)
            label = torch.norm(x_hat - x.unsqueeze(dim=0), dim=-1).argmin(dim=0)

        x_reconstruct = []
        for i in range(x.shape[0]):
            x_reconstruct.append(self.D[label[i]] @ self.D[label[i]].T @ x[i])

        x_reconstruct = torch.stack(x_reconstruct)

        subspace_loss = F.mse_loss(x_reconstruct, x) * self.gamma1

        return x_reconstruct, subspace_loss, label

    def upadte_D(self, z, pred):
        with torch.no_grad():
            for i in range(self.cluster_num):
                z_temp = z[pred == i]
                z_reconstruct = z_temp @ (self.D[i] @ self.D[i].T)
                error = torch.norm(z_temp - z_reconstruct, dim=1)
                valid_indices = torch.argsort(error, descending=True)[:int(z.size(0)*0.5)]
                _, _, vh = torch.linalg.svd(z_temp[valid_indices], full_matrices=False)

                d = min(self.factor_num_per_cluster, vh.size(0))
                self.D[i,:,:d].copy_(vh[:d].T)
                self.D[i,:,d:].mul_(0)

class self_exp(nn.Module):


    def __init__(self, size=1000, gamma0=1.0, gamma1=0.1):
        super(self_exp, self).__init__()
        self.size = size
        self.C = nn.Parameter(torch.randn((size, size))/ size**0.5 )
        self.C.data.fill_diagonal_(0)
        mask = torch.ones((size, size))
        mask.fill_diagonal_(0)
        self.register_buffer('mask', mask)

        self.gamma0 = gamma0
        self.gamma1 = gamma1

    def forward(self, x):

        x_reconstruct = (self.C * self.mask) @ x 

        reconstruct_loss = torch.sum((self.C * self.mask).square()) * self.gamma0 / 2 + \
            F.mse_loss(x_reconstruct, x, reduction='sum') * self.gamma1 / 2
        
        return x_reconstruct, reconstruct_loss, None


class SCNet(nn.Module):
    """
    This is a "foo" model, i.e., just an example of pytorch model.
    """

    def __init__(self, encoder, decoder, cluster_module, pretrain=False):
        super(SCNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cluster_m = cluster_module
        self.pretrain = pretrain

    def forward(self, x, intermediate=False):
        
        z = self.encoder(x)
        N, C, H, W = z.size()
        z = z.view(N, -1)
        if self.pretrain:
            x_recostruct = self.decoder(z.view(N, C, H, W))
            self_exp_loss, preds = None, None
        else:
            z_reconstruct, self_exp_loss, preds = self.cluster_m(z)
            x_recostruct = self.decoder(z_reconstruct.view(N, C, H, W))

        if not intermediate:
            return x_recostruct, self_exp_loss, preds, None
        else:
            return x_recostruct, self_exp_loss, preds, z


def create_ae(kernels, channels):

    encoder_modules = []
    for i in range(1, len(channels)):
        #  Each layer will divide the size of feature map by 2
        encoder_modules.append(nn.Conv2d(channels[i-1], channels[i], kernels[i-1], stride=2, padding=kernels[i-1]//2))
        encoder_modules.append(nn.ReLU(True))
    encoder = nn.Sequential(*encoder_modules)

    decoder_modules = []
    channels = list(reversed(channels))
    kernels = list(reversed(kernels))
    for i in range(1, len(channels)):
        #  Each layer will divide the size of feature map by 2
        decoder_modules.append(nn.ConvTranspose2d(channels[i-1], channels[i], kernels[i-1], stride=2, padding=(kernels[i-1]-1)//2, output_padding=1))
        decoder_modules.append(nn.ReLU(True))
    decoder = nn.Sequential(*decoder_modules) 

    return encoder, decoder