"""
The main file to run experiments.
"""
import argparse
import logging
import os

import torch
from transformers import AutoTokenizer
import numpy as np

from data import get_GLUE_datasets
from model import TdscLanguageModel
from train import train
from util import set_random_seeds
from unsup_baseline import unsup_baseline

def options():
    """
    add command line options here
    """
    parser = argparse.ArgumentParser()
    # yapf: disable
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
    parser.add_argument('--unsup_losses_weights', type=str, default='100|1|1',
        help='weights are gamma_0, _1, _2 in Table 3 of the TSDC paper')
    parser.add_argument('--unsup_loss_final_weight', type=float, default=1.,
                        help='the final weight of unsupervised loss vs supervised loss')
    parser.add_argument('--get_unsupervised_baseline', action='store_true')

    # data / io related
    parser.add_argument('--dataset_name', type=str, default='sst2')
    parser.add_argument('--checkpoint_dir', type=str, default='saved_models/best.pth',
                        help='where to save the model')
    parser.add_argument('--load_model_from', type=str, help='load previous model')
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
    parser.add_argument('--linear_decay', action='store_true', help='use linear deacy scheduler')
    # model
    parser.add_argument('--classifier_dropout', type=float, default=0.1,
                        help='dropout prob of the classifier MLP')
    # wandb
    parser.add_argument('--wandb', action='store_true', help='use wandb for logs')
    parser.add_argument('--wandb_runname', type=str, help='experiment specific name')
    parser.add_argument('--log_every', type=int, default=0,
                        help='log training process to wandb every k steps')

    # extra
    parser.add_argument('--seed', type=int, default=2022, help="the random seed")
    # yapf: enable
    args = parser.parse_args()
    return args


def main():
    args = options()
    set_random_seeds(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_path = os.path.dirname(args.checkpoint_dir)
    os.makedirs(base_path, exist_ok=True)

    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(base_path, 'train_log.log'), mode='a'),
            logging.StreamHandler(),
        ],
        format='%(asctime)s:%(msecs)d|%(name)s|%(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO)
    logging.info('Start Training!')

    # load model, tokenizer
    model = TdscLanguageModel(args).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.load_model_from and os.path.isfile(args.load_model_from):
        model.load_state_dict(torch.load(args.load_model_from), strict=True)

    # load data
    # NOTE: by passing tokenizer into this function, the 'text' field is now
    # tokenized text.
    data = get_GLUE_datasets(dataset_name=args.dataset_name,
                             tokenizer=tokenizer,
                             max_seq_length=args.max_seq_length,
                             seed=args.seed)

    # NOTE by Jiayang: The following is unnecessary anymore, since the new data.py
    # takes care of the preprocessing, so that sst2, sst5, and mnli-mismatched have
    # train/valid/test split properly set up.
    # if 'valid' not in data.keys() and 'validation' in data.keys():
    #     data['valid'] = data.pop('validation')
    # if 'test' not in data.keys() or args.dataset_name == 'sst2':
    #     data['test'] = data['valid']
    if args.get_unsupervised_baseline:
        test_accuracy, train_accuracy, hidden_states, cluster_label = unsup_baseline(model, data, args, device=device)
        logging.info(f'unsupervised baseline (spectral clustering): '
                     f'test set: {test_accuracy} | train set {train_accuracy}')
        np.save(os.path.join(base_path, 'hidden_states.npy'), hidden_states)
        np.save(os.path.join(base_path, 'cluster_label.npy'), cluster_label)
        return
    # TODO: add more init & control here
    loss_fn = torch.nn.CrossEntropyLoss()  # TODO: change loss_fn.
    train(model, data, loss_fn, args, device=device)


if __name__ == '__main__':
    main()
