"""
Contains the implementation of models
"""

from symbol import factor
from torch import nn
import torch


class Kfactor(nn.Module):
    """
    SelfExpression module, implement K-FACTORIZATION SUBSPACE CLUSTERING introduced in https://dl.acm.org/doi/pdf/10.1145/3447548.3467267
    """

    def __init__(self, cluster_num, factor_dim=128, factor_num_per_cluster=64):
        super().__init__()
        assert factor_dim > factor_num_per_cluster

        self.cluster_num = cluster_num
        self.factor_dim = factor_dim
        self.factor_num_per_cluster = factor_num_per_cluster

        self.D = nn.Parameter(torch.randn((cluster_num, factor_dim, factor_num_per_cluster)))


    def forward(self, x):
        """
        Compute C given D, this process is not involed in backward propagation
        """
        with torch.no_grad():
            Dtx = torch.einsum('nij,bi->nbj', self.D, x)
            Cs = torch.einsum('nbj,nkj->nbk', Dtx, self.DTD_invs)
            x_hat = torch.einsum('nij,nbj->nbi', self.D, Cs)
            label = torch.norm(x_hat - x.view(1, -1, self.factor_dim), dim=-1).argmin(dim=0)

            C = Cs[label, [i for i in range(x.shape[0])]]


        return C, label

    def upadte_DTD_inv(self):
        with torch.no_grad():
            DTD_inv_list = []
            for d in self.D:
                dtd = d.T @ d
                DTD_inv_list.append(torch.linalg.inv(dtd))

            self.DTD_invs = torch.stack(DTD_inv_list, dim=0)


    def compute_subsapce_loss(self, x, C, label):

        loss = 0
        for i in range(x.shape[0]):

            loss += torch.sum((x[i]- self.D[label[i]] @ C[i]).square())

        return loss


class FooModel(nn.Module):
    """
    This is a "foo" model, i.e., just an example of pytorch model.
    """

    def __init__(self, encoder, decoder, factor_dim=128):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.self_exp = Kfactor(10, factor_dim=factor_dim, factor_num_per_cluster=16)
        self.self_exp.upadte_DTD_inv()

    def forward(self, x):

        z = self.encoder(x)
        x_hat = self.decoder(z)
        C, label = self.self_exp(z)
        subspace_loss = self.self_exp.compute_subsapce_loss(z, C, label)

        return x_hat, label, subspace_loss


def create_MLP(input_dim, output_dim, latent_dim, layer_num, add_BN=True):

    layers = [nn.Linear(input_dim, latent_dim)]
    if add_BN:
        layers.append(nn.BatchNorm1d(latent_dim))
    layers.append(nn.ReLU())
    for _ in range(layer_num):
        layers.append(nn.Linear(latent_dim, latent_dim))
        if add_BN:
            layers.append(nn.BatchNorm1d(latent_dim)) 
        layers.append(nn.ReLU())   
    layers.append(nn.Linear(latent_dim, output_dim))

    model = nn.Sequential(*layers)

    return model

    