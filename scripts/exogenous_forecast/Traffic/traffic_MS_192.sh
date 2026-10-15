#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ../Time-Series-Library/dataset/traffic/ --data_path traffic.csv --data custom --model ExoPath --model_id traffic_MS_96_192 --features MS --target OT --seq_len 96 --label_len 48 --pred_len 192 --enc_in 862 --dec_in 862 --c_out 1 --learning_rate 0.0005 --weight_decay 1e-07 --lradj cosine --patience 15 --d_model 320 --t_ff 1536 --num_groups 256 --batch_size 8 --dropout 0.1 --dbloss_alpha 0.95 --dbloss_beta 0.35 --train_epochs 30 --use_dbloss 1 --des exp --gpu 0 --use_gpu 1 --seed 2025
