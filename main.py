"""
The main file to run experiments.
"""

import os
import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer

from util import set_random_seeds
from data import get_MNIST_datasets, get_MNIST_dataloaders
from train import train
from model import TdscLanguageModel


def options():
    """
    add command line options here
    """
    parser = argparse.ArgumentParser()
    # add arguments
    # model related
    # TODO: scan through this repo to find args._model_related_flag_

    # training / optimization related
    parser.add_argument('--n_epochs', type=int, default=10, help='number of times to train')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
    parser.add_argument('--decay_steps',
                        type=int,
                        default=0,
                        help='decay lr after x epochs. 0 means to use ReduceLrOnPlateau')
    parser.add_argument('--lr_decay_rate', type=float, default=0.8, help='how much to decay lr')
    parser.add_argument('--patience',
                        type=int,
                        default=3,
                        help='how many epochs to wait until reducing lr; used in ReduceLROnPlateau')
    # extra
    parser.add_argument('--seed', type=int, default=2022, help="the random seed")
    parser.add_argument('--verbose', action='store_true', help='whether to print results a lot')
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default='saved_models/best.pth',
                        help='where to save the model')

    args = parser.parse_args()
    return args


def main():
    args = options()
    set_random_seeds(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model, tokenizer
    model = TdscLanguageModel(args=args).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # load data
    # process data to a dict {'input_ids': input_ids (for all training sample)}
    # so that we can form the triplet data easily, by calling train_data = [data[Da, Dp, Dn] for D.. in batches]
    # should keep 'text' and 'label' on each dataset
    tokenized_data = {}     # split: (data, label, n_samples)
    
    for split in ['train', 'valid', 'test']:
        data =  pd.read_csv(os.path.join(agrs.data_path, split+'.csv'))[['text','label']] \
        tokenized_data[split] = [
            tokenizer(data[split]['text'], 
                    padding=args.padding_strategy, 
                    max_length=args.max_seq_length, 
                    truncation=True),
            data[split]['label'],
            len(data[split]['label'])
        ]

    # TODO: add more init & control here
    loss_fn = torch.nn.CrossEntropyLoss()  # TODO: change loss_fn.
    train(model, tokenized_data, loss_fn, args, device=device, mode=['unsup'])


if __name__ == '__main__':
    main()
