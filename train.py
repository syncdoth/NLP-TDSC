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
          tokenized_data,
          loss_fn,
          args,
          device='cpu',
          training_modes=['unsup', 'nlp']):
    """ Trains a given model and dataset.

    obtained and adapted from:
    https://github.com/fabio-deep/Distributed-Pytorch-Boilerplate/blob/master/src/train.py
    """

    ''' training_modes

    check training_modes only contains options from the following:
    'unsup': unsupervised training (including triplet loss, self expression loss, ...),
        output can be used as classification result,
    'sup': usual supervised training, may call this mode after 'unsup' training to correct model's prediction
    'nlp': some NLP pretraining tasks, as substitutions for Reconstruction Task

    Note that if we only train on unsup mode, the model will collapse to constant function
    So we recommend to have 'sup' or 'nlp' in training_modes

    '''
    num_training_samples = tokenized_data['train'][2]

    if set(mode).difference(['unsup', 'sup', 'nlp']):
        raise NotImplementedError(f'Not support {set(mode).difference(['unsup', 'sup', 'nlp'])} training modes')
    if 'unsup' in mode:
        # initialize unsup labels to construct Triplet Dataset
        unsup_labels = torch.empty(num_training_samples, dtype=torch.int8).random_(args.num_unsup_clusters)
        latent_embs_for_clustering = []  # for clustering

    # TODO: load tokenized_data['valid'] and tokenized_data['test'] by dataloader
    # or modify evaluate() function

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

    index = np.shuffle(range(num_training_samples))
    # train_dataloader = tokenized_data['train'][0]
    num_steps_per_epoch = (num_training_samples-1)//args.batch_size + 1

    since = time.time()
    for epoch in range(args.n_epochs):
        # TODO: form training data by index list: index = np.shuffle(range(num_training_samples)), then split by batch
        # in training loop, just take data and labels by index
        if 'unsup' in mode:
            # TODO: call function implemented by Arman. D has 3 keys, ['anchor', 'pos', 'neg']
            # D['anchor'] should be range(num_training_samples)
            D = get_triplet_data(num_training_samples, unsup_labels)    

        model.train()
        sample_count = 0
        running_loss = 0
        running_acc = 0

        if args.verbose:
            logging.info(f'\nEpoch {epoch + 1}/{args.n_epochs}:\n')

        for i in tqdm(range(num_steps_per_epoch), position=0, total=num_steps_per_epoch):
            # beginning and ending index for this batch
            b, e = i*args.batch_size, min((i+1)*args.batch_size, num_training_samples)
            optimizer.zero_grad()

            # retrieve data
            if 'unsup' in mode:
                embs = {}
                loss = 0
                for s in ['anchor', 'pos', 'neg']:
                    idx = D[s][index[i]]    # index in this batch
                    input_ids = tokenized_data['train'][0]['input_ids'][idx].to(device)
                    attention_mask = tokenized_data['train'][0]['attention_mask'][idx].to(device)
                    embs[s] = model.get_lm_embedding(input_ids, attention_mask)
                    if s == 'anchor':
                        latent_embs_for_clustering.extend(np.array(embs))   # to make a non-gradient copy of embs

                    if 'sup' in mode:
                        labels = tokenized_data['train'][1][idx].to(device)
                        logits = model.classifier(embs[s])
                        sup_loss = loss_fn(logits, labels)
                        loss += sup_loss

                        sample_count += input_ids.size(0)
                        running_loss += loss.item() * input_ids.size(0)  # smaller batches count less
                        running_acc += (yhat.argmax(-1) == labels).sum().item()  # num corrects

                # unsup loss, including Triplet Loss and Self Expression Loss
                unsup_loss = get_unsup_loss(embs)
                loss += unsup_loss

                loss.backward()
                optimizer.step()

            if 'sup' in mode:
                idx = index[i]
                input_ids = tokenized_data['train'][0]['input_ids'][].to(device)
                attention_mask = tokenized_data['train'][0]['attention_mask'][idx].to(device)
                labels = tokenized_data['train'][1][idx].to(device)

                embs = model.get_lm_embedding(input_ids, attention_mask)
                logits = model.classifier(embs)

                loss = loss_fn(logits, labels)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()

                sample_count += input_ids.size(0)
                running_loss += loss.item() * input_ids.size(0)  # smaller batches count less
                running_acc += (yhat.argmax(-1) == labels).sum().item()  # num corrects

        epoch_train_loss = running_loss / sample_count
        epoch_train_acc = running_acc / sample_count

        # update unsup labels
        train_unsup_labels = update_unsup_labels(latent_embs_for_clustering, method='emb or affinity matrix?')
        latent_embs_for_clustering= []

        # reduce lr
        if args.decay_steps > 0:
            lr_decay.step()
        else:  # reduce on plateau, evaluate to keep track of acc in each process
            epoch_valid_loss, epoch_valid_acc = evaluate(model,
                                                         tokenized_data['valid'],
                                                         loss_fn,
                                                         args,
                                                         device=device)
            lr_decay.step(epoch_valid_acc[0])

        if args.verbose:  # only validate using process 0
            if epoch_valid_loss is None:  # check if process 0 already validated
                epoch_valid_loss, epoch_valid_acc = evaluate(model,
                                                             tokenized_data['valid'],
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

        test_loss, test_acc = evaluate(model, tokenized_data['test'], loss_fn, args, device=device)

        logging.info(f'\nBest [Valid] | epoch: {best_epoch} - loss: {best_valid_loss:.4f} '
                     f'- acc: {best_valid_acc:.4f}')
        logging.info(f'[Test] loss {test_loss:.4f} - acc: {test_acc:.4f}')
