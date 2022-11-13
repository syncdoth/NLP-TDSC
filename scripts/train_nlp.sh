# setup your experiment name; it will be used to determine log directory name and
# wandb log name
exp_name=roberta_base_sst2_joint_linear_lr1e-5_w1e-4

python main.py \
    --model_name_or_path roberta-base \
    --dataset_name sst2 \
    --num_sup_labels 2 \
    --num_unsup_clusters 2 \
    --num_factor_per_cluster 256 \
    --unsup_clustering_method kfactor \
    --training_modes 'sup|unsup' \
    --max_seq_length 48 \
    --n_epochs 5 \
    --batch_size 128 \
    --optimizer adam \
    --weight_decay 0 \
    --linear_decay \
    --lr 1e-5 \
    --seed 2022 \
    --checkpoint_dir "saved_models/$exp_name/best.pth" \
    --verbose \
    --wandb \
    --wandb_runname $exp_name \
    --log_every 10 \
    --unsup_loss_final_weight 1e-4
