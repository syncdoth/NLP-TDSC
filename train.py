"""
implementation of the training loop.
"""
import gc
import logging
import time
from typing import Dict

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tqdm import tqdm
from util import acc, spectral_clustering, save_model
from sklearn.cluster import KMeans

from IPython import embed
import os


def train(model,
          dataloaders,
          args,
          device='cpu'):
    """ Trains a given model and dataset.

    obtained and adapted from:
    https://github.com/fabio-deep/Distributed-Pytorch-Boilerplate/blob/master/src/train.py
    """

    # optimizers
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              weight_decay=args.weight_decay,
                              momentum=0.9)
    else:
        raise NotImplementedError(f'{args.optimizer} not setup.')

    # lr schedulers

    lr_decay = lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

    if not args.initialize and os.path.exists(f'{args.pretrain_weights}/{args.data_type}.ckpt'):
        state_dict = torch.load(
            f'{args.pretrain_weights}/{args.data_type}.ckpt', map_location=device
        )
        model.encoder.load_state_dict(state_dict['encoder'])
        model.decoder.load_state_dict(state_dict['decoder'])
        
        print('******Load pretrained model successfully!******')
    
    data_train = dataloaders['data']
    label_train = dataloaders['label']

    # initialize cluster_module
   

    z_rec = []
    with torch.no_grad():
        for i in range(0, args.data_num, args.batch_size):
            inputs = data_train[i:i+args.batch_size].unsqueeze(dim=1).to(device)
            inputs /= inputs.max()
            inputs = F.interpolate(inputs, size=(inputs.size(-1)//8) * 8, mode='bilinear', align_corners=True)
            z = model.encoder(inputs)
            z_rec.append(z.reshape(z.size(0), -1))
        
        Zs = torch.cat(z_rec, dim=0)

    if not args.initialize and args.sc_type == 'kfactor':

        kmeans = KMeans(n_clusters=args.cluster_num).fit(Zs.cpu().numpy())
        y_pred = kmeans.labels_
        # print(f'Acc at initialization: {acc(y_pred, label_train.numpy()):.03f}')

        y_pred = torch.Tensor(y_pred).to(device)
        model.cluster_m.upadte_D(Zs, y_pred)
        with torch.no_grad():
            _, _, preds2 = model.cluster_m(Zs)
        
        print(f'Acc after initialization: {acc(preds2.cpu().numpy(), label_train.numpy()):.03f}')

        predictions = preds2

    elif not args.initialize and args.sc_type == 'spectral': 

        optimizer_init_cluster_m = optim.Adam(
            model.cluster_m.parameters(), lr=args.lr) 
        
        for i in tqdm(range(1000)):
            _, loss, _ = model.cluster_m(Zs)
            optimizer_init_cluster_m.zero_grad()
            loss.backward()
            optimizer_init_cluster_m.step()

        C = (model.cluster_m.C * model.cluster_m.mask).data.cpu().numpy()
        y_pred = spectral_clustering(C, args.cluster_num, args.dim_subspace, args.alpha, args.rho)

        print(f'Acc after initialization: {acc(y_pred, label_train.numpy()):.03f}')

        predictions = torch.Tensor(y_pred).long()

    if args.loss_type == 'triplet':

        loss_func = torch.nn.TripletMarginLoss(margin=0.5, p=2.0, eps=1e-06)

    pbar = tqdm(total=args.n_epochs)
    for epoch in range(args.n_epochs):

        model.train()
        sample_count = 0
        running_loss = 0

        indices = torch.randperm(args.data_num)
        preds_rec = []

        for i in range(0, args.data_num, args.batch_size):
            batch_indices = indices[i:i+args.batch_size]
            inputs = data_train[batch_indices].unsqueeze(dim=1).to(device)
            inputs /= inputs.max()
            inputs = F.interpolate(inputs, size=(inputs.size(-1)//8) * 8, mode='bilinear', align_corners=True)
            labels = label_train[batch_indices]

            xhat, subspace_loss, _, z = model(inputs, args.loss_type == 'triplet')

            reconstruction_loss = F.mse_loss(xhat, inputs, reduction='sum')

            if args.initialize:
                loss = reconstruction_loss
            else:
                loss = reconstruction_loss + subspace_loss

            if args.loss_type == 'triplet':
                preds_batch = predictions[batch_indices]
                pos_indices = []
                neg_indices = []

                for i in range(inputs.size(0)):
                    pos_i = torch.nonzero(preds_batch == preds_batch[i])
                    neg_i = torch.nonzero(preds_batch != preds_batch[i])
                    pos_indices.append(pos_i[torch.randint(pos_i.shape[0], size=(1,))])
                    neg_indices.append(neg_i[torch.randint(neg_i.shape[0], size=(1,))])
                
                pos_z = z[pos_indices]
                neg_z = z[neg_indices]

                loss += loss_func(z, pos_z, neg_z) * args.gamma2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sample_count += inputs.size(0)
            running_loss += loss.item()

        epoch_train_loss = running_loss / sample_count

        if not args.initialize and args.sc_type == 'kfactor':
            z_rec = []
            preds_rec = []
            with torch.no_grad():
                for i in range(0, args.data_num, args.batch_size):
                    inputs = data_train[i:i+args.batch_size].unsqueeze(dim=1).to(device)
                    inputs /= inputs.max()
                    inputs = F.interpolate(inputs, size=(inputs.size(-1)//8) * 8, mode='bilinear', align_corners=True)
                    z = model.encoder(inputs)
                    z_rec.append(z.reshape(inputs.size(0), -1))
                    _, _, preds = model.cluster_m(z.view(inputs.size(0), -1))
                    preds_rec.append(preds)

                Zs = torch.cat(z_rec, dim=0)
                Preds = torch.cat(preds_rec, dim=0)

                model.cluster_m.upadte_D(Zs, Preds)
                pbar.set_description(f"training loss:{epoch_train_loss:.04f}, acc: {acc(Preds.cpu().numpy(), label_train.numpy()):.03f}")
        else:
            pbar.set_description(f"training loss:{epoch_train_loss:.04f}")

        lr_decay.step()
        pbar.update()

        if epoch % args.screen_epoch == 0:
            if args.initialize:
                save_model(model, f'{args.pretrain_weights}/{args.data_type}.ckpt')
            else:
                save_model(model, f'{args.output_dir}/checkpoint.ckpt') 
        
        if not args.initialize and epoch % args.screen_epoch == args.screen_epoch-1 and args.sc_type == 'spectral':
        
            C = (model.cluster_m.C * model.cluster_m.mask).data.cpu().numpy()
            y_pred = spectral_clustering(C, args.cluster_num, args.dim_subspace, args.alpha, args.rho)

            print( f'Epoch: {epoch}, loss: {epoch_train_loss:.3f}, acc: {acc(y_pred, label_train.numpy()):.3f}.')

    save_model(model, f'{args.pretrain_weights}/{args.data_type}.ckpt')
    




