
# Laptop
CUDA_VISIBLE_DEVICES=0 python finetune.py --dataset Laptop --gnn attgcn --metric_type att --combination multi_conj\
        --model_name roberta-en --lr 1e-5 --layers 1 --attention_heads 1 --h_dim 60 --n_pass 3 --q_pass 3 --k_pass 3\
        --freq_type tkdft --result_file sigmoid_tkdft_roberta_base.json --probe_layers '-1' --pass_type bond --start_freq 8\
        --chpt_dir best_models --sparse_hold 0.5

# Rest
CUDA_VISIBLE_DEVICES=0 python finetune.py --dataset Restaurants --gnn attgcn --metric_type att --combination multi_conj\
        --model_name roberta-en --lr 1e-5 --layers 1 --attention_heads 1 --h_dim 60 --n_pass 22 --q_pass 22 --k_pass 22\
        --freq_type tkdft --result_file sigmoid_tkdft_roberta_base.json --probe_layers '-1' --pass_type high --start_freq 8\
        --chpt_dir best_models --sparse_hold 0.5


# Twitter
CUDA_VISIBLE_DEVICES=0 python finetune.py --dataset Tweets --gnn attgcn --metric_type att --combination multi_conj\
         --model_name roberta-en --lr 1e-5 --layers 1 --attention_heads 1 --h_dim 60 --n_pass 8 --q_pass 8 --k_pass 8\
          --freq_type tkdft --result_file sigmoid_tkdft_roberta_base.json --probe_layers '-1' --pass_type bond --start_freq 8\
          --chpt_dir best_models --sparse_hold 0.5

