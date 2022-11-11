# nlp-task training
python main.py \
    --model_name_or_path roberta-base \
    --dataset_name sst2 \
    --training_modes 'unsup|sup' \
    --max_seq_length 48 \
    --n_epochs 2 \
    --batch_size 64 \
    --optimizer adam \
    --weight_decay 0 \
    --lr 1e-5 \
    --linear_decay \
    --seed 2022 \
    --checkpoint_dir saved_models/roberta_base_sst2.pth \
    --verbose

# original tdsc training
python tdsc.py --db coil20 --show-freq 10 --seed 100