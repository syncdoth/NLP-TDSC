# TDSC-Net applied on NLP

This is a group project repo for Team 16, COMP 5331, KDD (2022 Fall), HKUST

## How to train

```bash

# run TDSCNET on mnist using kfactor

# First pretrain autoecoder by add --initialize

python3 main.py --initialize  --verbose --sc_type spectral  --n_epochs 200 --batch_size 500 --cluster_num 10 --data_type mnist --data_num 10000 --dim_subspace 12

# now train TDSC-Net

python3 main.py --verbose  --n_epochs 100 --batch_size 500 --cluster_num 10 --data_type mnist --sc_type kfactor --gamma1 0.1 --data_num 2000 --dim_subspace 12 --loss_type triplet
```