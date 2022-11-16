#!/bin/bash

# python3 main.py --initialize  --verbose --sc_type spectral  --n_epochs 200 --batch_size 500 --cluster_num 10 --data_type mnist --data_num 10000 --dim_subspace 12

# python3 main.py  --verbose --sc_type spectral  --n_epochs 400 --batch_size -1 --cluster_num 10 --data_type mnist --screen_epoch 40 --data_num 2000 --gamma0 10 --gamma1 10

# python3 main.py --initialize --verbose  --n_epochs 200 --batch_size 500 --cluster_num 10 --data_type mnist --sc_type kfactor --data_num -1 --dim_subspace 12

# python3 main.py --verbose  --n_epochs 100 --batch_size 500 --cluster_num 10 --data_type mnist --sc_type kfactor --gamma1 0.1 --data_num 50000 --dim_subspace 12


# python3 main.py  --verbose --data_type mnist --sc_type spectral --loss_type triplet --n_epochs 400 --batch_size -1 --cluster_num 10 --screen_epoch 40 --data_num 2000 --gamma0 100 --gamma1 100 --gamma2 0.01


python3 main.py --verbose  --n_epochs 100 --batch_size 500 --cluster_num 10 --data_type mnist --sc_type kfactor --gamma1 0.1 --data_num 2000 --dim_subspace 12 --loss_type triplet