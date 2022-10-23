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
from torch.optim import lr_scheduler
from tqdm import tqdm

from evaluate import evaluate

from IPython import embed
import os

from post_clustering import spectral_clustering, acc


def train(model: nn.Module,
          data,
          loss_fn,
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
    if args.decay_steps > 0:
        lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate)
    else:
        lr_decay = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                  factor=args.lr_decay_rate,
                                                  mode='max',
                                                  patience=30,
                                                  cooldown=20,
                                                  min_lr=1e-6,
                                                  verbose=True)

    best_valid_loss = np.inf
    best_valid_acc = 0

    since = time.time()



    train_data = data['train'].train_data.reshape(-1, 28*28) / 255

    perm = torch.randperm(train_data.shape[0])
    index = perm[:args.num_samples]
    inputs = train_data[index].float().to(device)
    labels = data['train'].train_labels[index].numpy()

    for epoch in tqdm(range(args.n_epochs)):
        model.train()
        sample_count = 0
        running_loss = 0
        running_acc = 0

        optimizer.zero_grad()
        xhat, subspace_loss = model(inputs)
        reconstruction_loss = loss_fn(xhat, inputs)
        loss = reconstruction_loss + subspace_loss * args.subspace_loss_lambda
        loss.backward()
        optimizer.step()
        lr_decay.step()

        if epoch % 100 == 0:
            C = model.self_exp.C.data.cpu().numpy()
            y_pred = spectral_clustering(C, 10, 16, 0.04, 8)
            accuracy = acc(labels, y_pred)
            print( f'\n[Train] loss: {loss.item():.4f} | Acc: {accuracy:.4f}')

        # if args.verbose:  # only validate using process 0
        #     if epoch_valid_loss is None:  # check if process 0 already validated
        #         epoch_valid_loss, epoch_valid_acc = evaluate(model,
        #                                                      dataloaders['valid'],
        #                                                      loss_fn,
        #                                                      args,
        #                                                      device=device)

        #     logging.info(
        #         f'\n[Train] loss: {epoch_train_loss:.4f} - acc: {epoch_train_acc:.4f} | [Valid] loss: {epoch_valid_loss:.4f} - acc: {epoch_valid_acc:.4f}')

            # save model and early stopping
            # if epoch_valid_acc >= best_valid_acc:
            #     best_epoch = epoch + 1
            #     best_valid_acc = epoch_valid_acc
            #     best_valid_loss = epoch_valid_loss
            #     # saving using process (rank) 0 only as all processes are in sync
            #     torch.save(model.state_dict(), args.checkpoint_dir)
            # epoch_valid_loss = None  # reset loss

        gc.collect()  # release unreferenced memory

    # if args.verbose:
    #     time_elapsed = time.time() - since
    #     logging.info(f'\nTraining time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    #     model.load_state_dict(torch.load(args.checkpoint_dir))  # load best model

    #     test_loss, test_acc = evaluate(model, dataloaders['test'], loss_fn, args, device=device)

    #     logging.info(f'\nBest [Valid] | epoch: {best_epoch} - loss: {best_valid_loss:.4f} - acc: {best_valid_acc:.4f}')
    #     logging.info(f'[Test] loss {test_loss:.4f} - acc: {test_acc:.4f}')

