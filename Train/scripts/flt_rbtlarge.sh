# laptop

CUDA_VISIBLE_DEVICES=0 python finetune.py --dataset Laptop --gnn attgcn --metric_type att --combination multi_conj\
        --model_name roberta-en-large --lr 8e-6 --layers 1 --attention_heads 1 --h_dim 60 --n_pass 10 --q_pass 10 --k_pass 10\
        --freq_type tkdft --result_file mscale/best_tkdft_roberta_large_differ_seed.json --probe_layers '-1' --pass_type high --start_freq 8\
        --chpt_dir rbtlarge_best_models --sparse_hold 0.5 --seed ${freq}

# Rest
CUDA_VISIBLE_DEVICES=0 python finetune.py --dataset Restaurants --gnn attgcn --metric_type att --combination multi_conj\
        --model_name roberta-en-large --lr 1e-5 --layers 1 --attention_heads 1 --h_dim 60 --n_pass 5 --q_pass 5 --k_pass 5\
        --freq_type tkdft --result_file mscale/best_tkdft_roberta_large_differ_seed.json --probe_layers '-1' --pass_type high --start_freq 8\
        --chpt_dir rbtlarge_best_models --sparse_hold 0.5 --seed ${freq}


# Twitter
CUDA_VISIBLE_DEVICES=0 python finetune.py --dataset Tweets --gnn attgcn --metric_type att --combination multi_conj\
        --model_name roberta-en-large --lr 1e-5 --layers 1 --attention_heads 1 --h_dim 60 --n_pass 8 --q_pass 8 --k_pass 8\
        --freq_type tkdft --result_file mscale/best_tkdft_roberta_large_differ_seed.json --probe_layers '-1' --pass_type bond --start_freq 8\
        --chpt_dir rbtlarge_best_models --sparse_hold 0.5 --seed ${freq}



