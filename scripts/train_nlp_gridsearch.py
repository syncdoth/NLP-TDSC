# coding: utf-8

import os

def get_runstr_combinations(header='python3 finetune_lm.py', fixed_configs={}, comb_configs={}):
    """ Get grid search running scripts.
    """
    from itertools import product

    all_runstrs = []

    # add fixed arguments
    string = header
    for key in fixed_configs:
        string += ' --'+key+' '+str(fixed_configs[key])

    # add all combinations
    key_list = list(comb_configs.keys())
    value_list = [comb_configs[key] for key in key_list]
    for vals in list(product(*value_list)):
        tmp_str = string
        for key, val in zip(key_list, vals):
            tmp_str += ' --'+key+' '+str(val)
        all_runstrs.append(tmp_str)
    
    return all_runstrs

if __name__ == '__main__':
    header = "python3 main.py --verbose --wandb"
    fixed_configs = {
        'model_name_or_path': 'roberta-base',
        'dataset_name': 'sst2', 
        'num_sup_labels': 2,
        'num_unsup_clusters': 2,
        'num_factor_per_cluster': 256,
        'unsup_cluster_method': 'kfactor',
        'training_modes': 'unsup',
        'max_seq_length': 48,
        'n_epochs': 2,
        'batch_size': 128,
        'optimizer': 'adam',
        'weight_decay': 0,
        'seed': 2022,
        'checkpoint_dir': f'saved_models/{os.environ['exp_name']}/best.pth',
        'wandb_runname': os.environ['exp_name'],
        'log_every': 100,
    }
    comb_configs = {
        'lr': [1e-3, 5e-4, 1e-4, 5e-5],
    }

    all_runstrs = get_runstr_combinations(header, fixed_configs, comb_configs)
    print(all_runstrs)

    # # Start executing the run strings
    # for i, runstr in enumerate(all_runstrs):
    #     print(f'[{i+1}/{len(all_runstrs)}]', runstr)
    #     os.system(runstr)

    