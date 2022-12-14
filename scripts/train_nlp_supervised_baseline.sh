# setup your experiment name; it will be used to determin log directory name and
# wandb log name
exp_name=roberta_base_sst2_supervised_baseline

python main.py \
    --model_name_or_path roberta-base \
    --dataset_name sst2 \
    --num_sup_labels 2 \
    --training_modes 'sup' \
    --max_seq_length 48 \
    --n_epochs 5 \
    --batch_size 64 \
    --optimizer adam \
    --linear_decay \
    --weight_decay 0 \
    --lr 1e-5 \
    --seed 2022 \
    --checkpoint_dir "saved_models/$exp_name/best.pth" \
    --verbose \
    --wandb --wandb_runname $exp_name \
    --log_every 10


