"""
This file contains utility functions.
"""

import os
import random
from typing import Union, List, Tuple

import numpy as np
import torch
from torch import nn

import numpy as np
from sklearn import cluster
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize


def set_random_seeds(seed):
    """
    set the random seed of all related libraries
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def freeze_net(module: nn.Module, keys: Union[List, Tuple] = None):
    """
    freeze (don't allow training) the weights of a pytorch nn.Module
    """
    for k, p in module.named_parameters():
        if keys is None or k in keys:
            p.requires_grad = False


def unfreeze_net(module: nn.Module):
    """
    un-freeze (allow training) the weights of a pytorch nn.Module
    """
    for p in module.parameters():
        p.requires_grad = True


if __name__ == '__main__':
    test_layer = nn.Linear(1, 1)
    freeze_net(test_layer, keys=['weight'])
    print([
        f'{k} requires_grad: {p.requires_grad}' for k, p in test_layer.named_parameters()
    ])
    unfreeze_net(test_layer)
    print([
        f'{k} requires_grad: {p.requires_grad}' for k, p in test_layer.named_parameters()
    ])


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind_row, ind_col = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size


def thrC(C, alpha):
    if alpha < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > alpha * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp


def post_proC(C, K, d, ro):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    n = C.shape[0]
    C = 0.5 * (C + C.T)
    C = C - np.diag(np.diag(C)) + np.eye(n, n)  # good for coil20, bad for orl
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(n))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** ro)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L)
    return grp, L


def spectral_clustering(C, K, d, alpha, ro):
    C = thrC(C, alpha)
    y, _ = post_proC(C, K, d, ro)
    return y

def save_model(model, model_dir):

    state_dict = {
        'encoder': model.encoder.state_dict(),
        'decoder': model.decoder.state_dict(),
        'cluster_m': model.cluster_m.state_dict()
    }

    torch.save(state_dict, model_dir)