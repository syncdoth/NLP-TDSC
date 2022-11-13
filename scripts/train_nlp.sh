# setup your experiment name; it will be used to determine log directory name and
# wandb log name
exp_name=roberta_base_sst2_unsup

python main.py \
    --model_name_or_path roberta-base \
    --dataset_name sst2 \
    --num_sup_labels 2 \
    --num_unsup_clusters 2 \
    --num_factor_per_cluster 256 \
    --unsup_clustering_method kfactor \
    --training_modes 'unsup' \
    --max_seq_length 48 \
    --n_epochs 2 \
    --batch_size 128 \
    --optimizer adam \
    --weight_decay 0 \
    --lr 1e-3 \
    --seed 2022 \
    --checkpoint_dir "saved_models/$exp_name/best.pth" \
    --verbose \
    --wandb \
    --wandb_runname $exp_name \
    --log_every 50
    # --linear_decay
