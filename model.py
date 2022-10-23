"""
Contains the implementation of models
"""

from symbol import factor
from torch import nn
import torch
from IPython import embed

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


class SelfExpression(nn.Module):

    def __init__(self, num_samples):

        super().__init__()

        self.C = nn.Parameter(1.0e-8 * torch.ones(num_samples, num_samples, dtype=torch.float32), requires_grad=True)

    def forward(self, x):

        y = self.C @ x - torch.diag(self.C).reshape(-1, 1) * x

        return y


class FooModel(nn.Module):
    """
    This is a "foo" model, i.e., just an example of pytorch model.
    """

    def __init__(self, encoder, decoder, num_samples):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.self_exp = SelfExpression(num_samples)

    def forward(self, x):

        z = self.encoder(x)
        x_hat = self.decoder(z)
        z_hat = self.self_exp(z)

        subspace_loss = (z-z_hat).square().sum(dim=1).mean()

        return x_hat, subspace_loss


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

    