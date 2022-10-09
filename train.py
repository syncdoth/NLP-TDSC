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


def train(model: nn.Module,
          dataloaders: Dict[str, torch.utils.data.DataLoader],
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
    patience_counter = 0

    since = time.time()
    for epoch in range(args.n_epochs):

        model.train()
        sample_count = 0
        running_loss = 0
        running_acc = 0

        if args.verbose:
            logging.info(f'\nEpoch {epoch + 1}/{args.n_epochs}:\n')
            # tqdm for process (rank) 0 only when using distributed training
            train_dataloader = tqdm(dataloaders['train'])
        else:
            train_dataloader = dataloaders['train']

        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            yhat = model(inputs)

            loss = loss_fn(yhat, labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            sample_count += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)  # smaller batches count less
            running_acc += (yhat.argmax(-1) == labels).sum().item()  # num corrects

        epoch_train_loss = running_loss / sample_count
        epoch_train_acc = running_acc / sample_count

        # reduce lr
        if args.decay_steps > 0:
            lr_decay.step()
        else:  # reduce on plateau, evaluate to keep track of acc in each process
            epoch_valid_loss, epoch_valid_acc = evaluate(model,
                                                         dataloaders['valid'],
                                                         loss_fn,
                                                         args,
                                                         device=device)
            lr_decay.step(epoch_valid_acc[0])

        if args.verbose:  # only validate using process 0
            if epoch_valid_loss is None:  # check if process 0 already validated
                epoch_valid_loss, epoch_valid_acc = evaluate(model,
                                                             dataloaders['valid'],
                                                             loss_fn,
                                                             args,
                                                             device=device)

            logging.info(f'\n[Train] loss: {epoch_train_loss:.4f} - acc: {epoch_train_acc:.4f} |'
                         f' [Valid] loss: {epoch_valid_loss:.4f} - acc: {epoch_valid_acc:.4f}')

            # save model and early stopping
            if epoch_valid_acc >= best_valid_acc:
                patience_counter = 0
                best_epoch = epoch + 1
                best_valid_acc = epoch_valid_acc
                best_valid_loss = epoch_valid_loss
                # saving using process (rank) 0 only as all processes are in sync
                torch.save(model.state_dict(), args.checkpoint_dir)
            else:
                patience_counter += 1
                if patience_counter == (args.patience - 10):
                    logging.info(f'\nPatience counter {patience_counter}/{args.patience}.')
                elif patience_counter == args.patience:
                    logging.info(
                        f'\nEarly stopping... no improvement after {args.patience} Epochs.')
                    break
            epoch_valid_loss = None  # reset loss

        gc.collect()  # release unreferenced memory

    if args.verbose:
        time_elapsed = time.time() - since
        logging.info(f'\nTraining time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        model.load_state_dict(torch.load(args.checkpoint_dir))  # load best model

        test_loss, test_acc = evaluate(model, dataloaders['test'], loss_fn, args, device=device)

        logging.info(f'\nBest [Valid] | epoch: {best_epoch} - loss: {best_valid_loss:.4f} '
                     f'- acc: {best_valid_acc:.4f}')
        logging.info(f'[Test] loss {test_loss:.4f} - acc: {test_acc:.4f}')
