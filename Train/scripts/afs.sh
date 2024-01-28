

# Restaurants
CUDA_VISIBLE_DEVICES=0 python finetune.py --dataset Restaurants --gnn attgcn --metric_type att --combination multi_conj\
        --model_name roberta-en --lr 1e-5 --layers 1 --attention_heads 1 --h_dim 60 --n_pass 10 --q_pass 10 --k_pass 10\
        --freq_type tkauto --result_file multiseed_tkauto_roberta_base_differ_seed.json --probe_layers '-1' --pass_type high --start_freq 8\
        --dist_threshold 0.65 --lr_for_selector 5e-3 --chpt_dir tkauto_best_rbtbase 


# Tweets
CUDA_VISIBLE_DEVICES=0 python finetune.py --dataset Tweets --gnn attgcn --metric_type att --combination multi_conj\
        --model_name roberta-en --lr 1e-5 --layers 1 --attention_heads 1 --h_dim 60 --n_pass 10 --q_pass 10 --k_pass 10\
        --freq_type tkauto --result_file multiseed_tkauto_roberta_base_differ_seed.json --probe_layers '-1' --pass_type high --start_freq 8\
        --dist_threshold 0.75 --lr_for_selector 5e-3 --chpt_dir tkauto_best_rbtbase 


# Laptop
CUDA_VISIBLE_DEVICES=0 python finetune.py --dataset Laptop --gnn attgcn --metric_type att --combination multi_conj\
        --model_name roberta-en --lr 1e-5 --layers 1 --attention_heads 1 --h_dim 60 --n_pass 10 --q_pass 10 --k_pass 10\
        --freq_type tkauto --result_file multiseed_tkauto_roberta_base_differ_seed.json --probe_layers '-1' --pass_type high --start_freq 8\
        --dist_threshold 0.75 --lr_for_selector 5e-3 --chpt_dir tkauto_best_rbtbase 

