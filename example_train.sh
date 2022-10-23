#/bin/bash

python main.py \
    --n_epochs 10 \
    --batch_size 128 \
    --optimizer adam \
    --weight_decay 0.01 \
    --lr 1e-3 \
    --decay_steps 0 \
    --lr_decay_rate 0.9 \
    --seed 2022 \
    --checkpoint_dir saved_models/example_train_best.pth \
    --verbose \
    --subspace_loss_lambda 0.01 \
    --decay_steps  1 \
