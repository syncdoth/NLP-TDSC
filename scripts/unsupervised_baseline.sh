exp_name="pure-unsupervised-baseline"
function unsup_baseline(){
    dataset=$1
    num_cluster=$2

    python main.py \
        --get_unsupervised_baseline \
        --model_name_or_path roberta-base \
        --dataset_name $dataset \
        --num_sup_labels $num_cluster \
        --num_unsup_clusters $num_cluster \
        --max_seq_length 48 \
        --batch_size 64 \
        --seed 2022 \
        --checkpoint_dir "saved_models/${exp_name}-${dataset}/best.pth"
}

# unsup_baseline sst2 2
# unsup_baseline sst5 5
unsup_baseline mnli 3

