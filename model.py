"""
Contains the implementation of models
"""

import torch
from torch import nn
import torch.nn.functional as F
import math


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
        # self.C = nn.Parameter(torch.randn((size, size))/ size**0.5)
        self.C = nn.Parameter(1e-8 * torch.ones((size, size))) 
        self.C.data.fill_diagonal_(0)
        mask = torch.ones((size, size))
        # mask.fill_diagonal_(0)
        self.register_buffer('mask', mask)

        self.gamma0 = gamma0
        self.gamma1 = gamma1

    def forward(self, x):

        x_reconstruct = (self.C * self.mask) @ x 

        reconstruct_loss = torch.sum((self.C * self.mask).square()) * self.gamma0 + \
            F.mse_loss(x_reconstruct, x, reduction='sum') * self.gamma1
        
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




class Conv2dSamePad(nn.Module):
    """
    Implement Tensorflow's 'SAME' padding mode in Conv2d.
    When an odd number, say `m`, of pixels are need to pad, Tensorflow will pad one more column at right or one more
    row at bottom. But Pytorch will pad `m+1` pixels, i.e., Pytorch always pads in both sides.
    So we can pad the tensor in the way of Tensorflow before call the Conv2d module.
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
    A tensor with width w_in, feed it to ConvTranspose2d(ci, co, kernel, stride), the width of output tensor T_nopad:
        w_nopad = (w_in - 1) * stride + kernel
    If we use padding, i.e., ConvTranspose2d(ci, co, kernel, stride, padding, output_padding), the width of T_pad:
        w_pad = (w_in - 1) * stride + kernel - (2*padding - output_padding) = w_nopad - (2*padding - output_padding)
    Yes, in ConvTranspose2d, more padding, the resulting tensor is smaller, i.e., the padding is actually deleting row/col.
    If `pad`=(2*padding - output_padding) is odd, Pytorch deletes more columns in the left, i.e., the first ceil(pad/2) and
    last `pad - ceil(pad/2)` columns of T_nopad are deleted to get T_pad.
    In contrast, Tensorflow deletes more columns in the right, i.e., the first floor(pad/2) and last `pad - floor(pad/2)`
    columns are deleted.
    For the height, Pytorch deletes more rows at top, while Tensorflow at bottom.
    In practice, we usually want `w_pad = w_in * stride`, i.e., the "SAME" padding mode in Tensorflow,
    so the number of columns to delete:
        pad = 2*padding - output_padding = kernel - stride
    We can solve the above equation and get:
        padding = ceil((kernel - stride)/2), and
        output_padding = 2*padding - (kernel - stride) which is either 1 or 0.
    But to get the same result with Tensorflow, we should delete values by ourselves instead of using padding and
    output_padding in ConvTranspose2d.
    To get there, we check the following conditions:
    If pad = kernel - stride is even, we can directly set padding=pad/2 and output_padding=0 in ConvTranspose2d.
    If pad = kernel - stride is odd, we can use ConvTranspose2d to get T_nopad, and then delete `pad` rows/columns by
    ourselves; or we can use ConvTranspose2d to delete `pad - 1` by setting `padding=(pad - 1) / 2` and `ouput_padding=0`
    and then delete the last row/column of the resulting tensor by ourselves.
    Here we implement the former case.
    This module should be called after the ConvTranspose2d module with shared kernel_size and stride values.
    And this module can only output a tensor with shape `stride * size_input`.
    A more flexible module can be found in `yaleb.py` which can output arbitrary size as specified.
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

def create_ae(kernels, channels):

    encoder_modules = []
    for i in range(1, len(channels)):
        #  Each layer will divide the size of feature map by 2
        encoder_modules.append(Conv2dSamePad(kernels[i - 1], 2))
        encoder_modules.append(nn.Conv2d(channels[i - 1], channels[i], kernel_size=kernels[i - 1], stride=2))
        encoder_modules.append(nn.ReLU(True))

    encoder = nn.Sequential(*encoder_modules)

    decoder_modules = []
    channels = list(reversed(channels))
    kernels = list(reversed(kernels))
    for i in range(len(channels)-1):
        #  Each layer will divide the size of feature map by 2
        decoder_modules.append(nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=kernels[i], stride=2))
        decoder_modules.append(ConvTranspose2dSamePad(kernels[i], 2))
        decoder_modules.append(nn.ReLU(True))
    decoder = nn.Sequential(*decoder_modules) 

    return encoder, decoder