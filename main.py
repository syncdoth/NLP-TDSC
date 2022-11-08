"""
The main file to run experiments.
"""

import os
import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer

from util import set_random_seeds
from data import *
from train import train
from evaluate import evaluate
from model import TdscLanguageModel


def options():
    """
    add command line options here
    """
    parser = argparse.ArgumentParser()

    # model / config related
    parser.add_argument('--model_name_or_path', type=str, default='roberta-base', 
        help='name of model to load from huggingface or path to pre-trained model/checkpoint')
    parser.add_argument('--max_seq_length', type=int, default=32, 
        help='depend on the length of training instance, set the optimal length for tokenized input')
    parser.add_argument('--num_sup_labels', type=int, default=2, 
        help='the number of classes in the classification problem')
    parser.add_argument('--num_unsup_clusters', type=int, default=8,
        help='the number of clusters used in deep space clustering methods, either KFactor or Spectral Clustering')
    parser.add_argument('--num_factor_per_cluster', type=int, default=64,
        help='the number of basis vectors in each cluster, regarding to KFactor clustering method')
    parser.add_argument('--unsup_clustering_method', type=str, default='kfactor', choices=['kfactor', 'spectral'])
    parser.add_argument('--training_modes', type=str, default='unsup|sup', 
        help='training modes to train the model on, including "sup" ~ supervised, "unsup" ~ unsupervised, '
            'and "nlp" ~ other pretraining tasks, e.g Mask Token, Next Sentence Prediction')

    # data / io related
    parser.add_argument('--dataset_name', type=str, default='sst2')
    parser.add_argument('--checkpoint_dir', type=str, default='saved_models/best.pth',
                        help='where to save the model')
    parser.add_argument('--verbose', action='store_true', help='whether to print results a lot')

    # training / optimization related
    parser.add_argument('--n_epochs', type=int, default=10, help='number of times to train')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
    parser.add_argument('--decay_steps', type=int,  default=0,
                        help='decay lr after x epochs. 0 means to use ReduceLrOnPlateau')
    parser.add_argument('--lr_decay_rate', type=float, default=0.8, help='how much to decay lr')
    parser.add_argument('--seed', type=int, default=2022, help="the random seed")

    args = parser.parse_args()
    return args


def main():
    args = options()
    set_random_seeds(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(os.path.dirname(args.checkpoint_dir), exist_ok=True)

    # load model, tokenizer
    model = TdscLanguageModel(args).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # load data
    # process data to a dict {'input_ids': input_ids (for all training sample)}
    # so that we can form the triplet data easily, by calling train_data = [data[Da, Dp, Dn] for D.. in batches]
    # should keep 'text' and 'label' on each dataset
    data = get_GLUE_datasets(args.dataset_name)
    # data = get_madeup_data_for_testing()
    print('Data split:', data.keys())
    if 'valid' not in data.keys() and 'validation' in data.keys():
        data['valid'] = data['validation']
    if 'test' not in data.keys():
        data['test'] = data['valid']
    tokenized_data = {}     # split: (data, label, n_samples)

    for split in ['train', 'valid', 'test']:
        tokenized_data[split] = [
            tokenizer(data[split]['text'],
                    padding='max_length',
                    max_length=args.max_seq_length,
                    truncation=True,
                    return_tensors='pt'),
            torch.tensor(data[split]['label']),
            len(data[split]['label'])
        ]

    # TODO: add more init & control here
    loss_fn = torch.nn.CrossEntropyLoss()  # TODO: change loss_fn.
    train(model, tokenized_data, loss_fn, args, device)


if __name__ == '__main__':
    main()
