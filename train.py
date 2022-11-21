"""
implementation of the training loop. column_width=120
"""
import gc
import logging
import time
from sklearn.cluster import SpectralClustering

import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from evaluate import evaluate
from data import get_triplet_data
from transformers import get_linear_schedule_with_warmup
from post_clustering import acc
import wandb


def train(model: nn.Module, tokenized_data, loss_fn, args, device='cpu'):
    """ Trains a given model and dataset.

    obtained and adapted from:
    https://github.com/fabio-deep/Distributed-Pytorch-Boilerplate/blob/master/src/train.py

    training_modes

    check training_modes only contains options from the following:
    'unsup': unsupervised training (including triplet loss, self expression loss, ...),
        output can be used as classification result,
    'sup': usual supervised training, may call this mode after 'unsup' training to correct model's prediction
    'nlp': some NLP pretraining tasks, as substitutions for Reconstruction Task

    Note that if we only train on unsup mode, the model will collapse to
        constant function ~ one of optimal solution of the training objective
    So we recommend to have 'sup' or 'nlp' in training_modes
    """
    num_training_samples = len(tokenized_data['train']['label'])
    training_modes = args.training_modes.split('|')

    if set(training_modes).difference(['unsup', 'sup', 'nlp']):
        raise NotImplementedError(
            f"Not support {set(training_modes).difference(['unsup', 'sup', 'nlp'])} training modes")
    if 'unsup' in training_modes:
        # initialize unsup labels to construct Triplet Dataset
        unsup_labels = torch.empty(num_training_samples,
                                   dtype=torch.int8).random_(args.num_unsup_clusters)

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

    # shuffle training data by index list, then in training loop, just take data and labels by batching index
    index = np.arange(num_training_samples)
    np.random.shuffle(index)
    # train_dataloader = tokenized_data['train'][0]
    num_steps_per_epoch = (num_training_samples - 1) // args.batch_size + 1

    # lr schedulers
    if args.linear_decay:
        lr_decay = get_linear_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=0,
                                                   num_training_steps=num_steps_per_epoch * args.n_epochs)
    elif args.decay_steps > 0:
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

    if args.wandb:
        if not args.wandb_runname:
            # args.wandb_runname = f'{args.mode}-{round(time.time() * 1000)}'
            args.wandb_runname = None
        experiment = wandb.init(entity='syncdoth',
                                project='NLP-TDSC',
                                name=args.wandb_runname,
                                config=args)
    since = time.time()
    for epoch in range(1, args.n_epochs + 1):
        if 'unsup' in training_modes:
            # create triplet dataset indices D. D has 3 keys, ['anchor', 'pos', 'neg']
            # D['anchor'] should be range(num_training_samples)
            if len(unsup_labels.unique()) == 1:
                logging.info("all points are in one cluster! Resetting the clusters..")
                unsup_labels = torch.empty(num_training_samples,
                                           dtype=torch.int8).random_(args.num_unsup_clusters)
            D = get_triplet_data(num_training_samples, unsup_labels)
            print('Unsup labels:\n', unsup_labels, '\nTriplet dataset:\n', D)
            embs_for_clustering = np.zeros(
                (num_training_samples, model.hidden_dim))  # for clustering
            kfactor_labels = torch.zeros(num_training_samples, dtype=torch.int8)

        model.train()
        sample_count = 1e-6
        running_loss = 0
        running_acc = 0
        unsup_running_acc = 0
        anchor_sample_count = 1e-6

        if args.verbose:
            logging.info(f'Epoch {epoch}/{args.n_epochs}:\n')

        for i in tqdm(range(num_steps_per_epoch),
                      position=0,
                      total=num_steps_per_epoch,
                      desc=f'Epoch {epoch}'):
            # beginning and ending index for this batch
            b, e = i * args.batch_size, min((i + 1) * args.batch_size, num_training_samples)
            optimizer.zero_grad()

            # retrieve data
            if 'unsup' in training_modes:
                embs = {}
                loss = 0
                for s in ['anchor', 'pos', 'neg']:
                    idx = D[s][index[b:e]]  # index in this batch
                    input_ids = tokenized_data['train']['text']['input_ids'][idx].to(device)
                    attention_mask = tokenized_data['train']['text']['attention_mask'][idx].to(device)
                    labels = tokenized_data['train']['label'][idx].to(device)
                    embs[s] = model.get_lm_embedding(input_ids, attention_mask)
                    if s == 'anchor':
                        embs_for_clustering[idx] = embs[s].cpu().detach().numpy()  # a non-gradient copy of embs

                    if 'sup' in training_modes and s == 'anchor':  # TODO: may train for 'anchor' only
                        logits = model.classifier(embs[s])
                        sup_loss = loss_fn(logits, labels)
                        loss += sup_loss

                        sample_count += input_ids.size(0)
                        running_loss += loss.item() * input_ids.size(0)  # smaller batches count less
                        running_acc += (logits.argmax(-1) == labels).sum().item()  # num corrects

                # unsup loss, including Triplet Loss and Self Expression Loss
                unsup_loss, kfactor_batch_label = model.get_unsup_loss(embs)
                kfactor_labels[D['anchor'][index[b:e]]] = kfactor_batch_label.cpu().detach().to(torch.int8)

                # since acc function divides by sample size, multiply it back so that
                # it is really a "running" acc, i.e. the sum of correct instances.
                anchor_sample_count += input_ids.size(0)
                unsup_running_acc += acc(labels.cpu().numpy(), kfactor_batch_label.cpu().numpy()) * labels.shape[0]
                if 'sup' in training_modes:
                    # if we are training in unsup|sup mode, control the weight of
                    # unsupservised loss.
                    # total_loss = supervised_loss + C * unsupervised_loss
                    unsup_loss *= args.unsup_loss_final_weight
                loss += unsup_loss
                loss.backward()
                optimizer.step()

            elif 'sup' in training_modes:
                idx = index[b:e]
                input_ids = tokenized_data['train']['text']['input_ids'][idx].to(device)
                attention_mask = tokenized_data['train']['text']['attention_mask'][idx].to(device)
                labels = tokenized_data['train']['label'][idx].to(device)

                embs = model.get_lm_embedding(input_ids, attention_mask)
                logits = model.classifier(embs)

                loss = loss_fn(logits, labels)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()

                sample_count += input_ids.size(0)
                running_loss += loss.item() * input_ids.size(0)  # smaller batches count less
                running_acc += (logits.argmax(-1) == labels).sum().item()  # num corrects

            if args.log_every > 0 and i % args.log_every == 0 and args.wandb:
                if isinstance(lr_decay, lr_scheduler._LRScheduler) :
                    last_lr = lr_decay.get_last_lr()[0]
                else:
                    # NOTE: ReduceLROnPlateau do not have this function.
                    last_lr = args.lr
                log_items = {
                    "train running loss": running_loss / sample_count,
                    "LR": last_lr,
                }
                if 'unsup' in training_modes:
                    log_items['unsup_loss'] = unsup_loss
                    log_items['train clustering running acc'] = unsup_running_acc / anchor_sample_count
                if 'sup' in training_modes:
                    log_items['train running accuracy'] = running_acc / sample_count
                experiment.log(log_items)

            if args.linear_decay:
                lr_decay.step()

        epoch_train_loss = running_loss / sample_count
        epoch_train_acc = running_acc / sample_count
        epoch_unsup_acc = unsup_running_acc / anchor_sample_count

        # update unsup labels. For spectral clustering, do not normalize latent embs
        if 'unsup' in training_modes:
            if args.unsup_clustering_method == 'kfactor':
                unsup_labels = kfactor_labels.clone()
            elif args.unsup_clustering_method == 'spectral':
                clustering = SpectralClustering(n_clusters=args.num_unsup_clusters,
                                                assign_labels='cluster_qr',
                                                random_state=args.seed)
                clustering.fit(embs_for_clustering)
                unsup_labels = torch.tensor(clustering.labels_)
                # TODO: compute clustering acc for spectral
            else:
                raise NotImplementedError(
                    '--unsup_clustering_method should be one of `kfactor` or `spectral`.')

        epoch_valid_loss, epoch_valid_acc = evaluate(model,
                                                     tokenized_data['valid'],
                                                     loss_fn,
                                                     args,
                                                     device=device)
        # reduce lr after epoch
        if args.decay_steps > 0:
            lr_decay.step()
        else:  # reduce on plateau, evaluate to keep track of acc in each process
            lr_decay.step(epoch_valid_acc)

        if args.verbose:  # only validate using process 0
            logging.info(f'[Train] loss: {epoch_train_loss:.4f} - acc: {epoch_train_acc:.4f} |'
                         f' clustering: {epoch_unsup_acc:.4f} |'
                         f' [Valid] loss: {epoch_valid_loss:.4f} - acc: {epoch_valid_acc:.4f}')
        if args.wandb:
            experiment.log({
                'valid loss': epoch_valid_loss,
                'valid accuracy': epoch_valid_acc,
                'epoch': epoch,
            })

        # save model and early stopping
        if epoch_valid_acc >= best_valid_acc:
            best_epoch = epoch
            best_valid_acc = epoch_valid_acc
            best_valid_loss = epoch_valid_loss
            # saving using process (rank) 0 only as all processes are in sync
            torch.save(model.state_dict(), args.checkpoint_dir)

        gc.collect()  # release unreferenced memory

    if args.verbose:
        time_elapsed = time.time() - since
        logging.info(f'Training time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        model.load_state_dict(torch.load(args.checkpoint_dir))  # load best model

        test_loss, test_acc = evaluate(model, tokenized_data['test'], loss_fn, args, device=device)

        logging.info(f'Best [Valid] | epoch: {best_epoch} - loss: {best_valid_loss:.4f} '
                     f'- acc: {best_valid_acc:.4f}')
        logging.info(f'[Test] loss {test_loss:.4f} - acc: {test_acc:.4f}')

        # add result metrics to the summary
        experiment.summary['best_valid_loss'] = best_valid_loss
        experiment.summary['best_valid_acc'] = best_valid_acc
        experiment.summary['test_loss'] = test_loss
        experiment.summary['test_acc'] = test_acc

    if args.wandb:
        wandb.finish()
