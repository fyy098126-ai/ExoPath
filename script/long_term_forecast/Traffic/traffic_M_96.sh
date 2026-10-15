#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ../Time-Series-Library/dataset/traffic/ --data_path traffic.csv --data custom --model ExoPath --model_id traffic_M_96_96 --features M --target OT --seq_len 96 --label_len 48 --pred_len 96 --enc_in 862 --dec_in 862 --c_out 862 --learning_rate 0.005 --weight_decay 0.0 --lradj cosine --patience 3 --d_model 448 --t_ff 128 --num_groups 128 --batch_size 48 --dropout 0.15 --dbloss_alpha 0.1 --dbloss_beta 0.4 --train_epochs 30 --use_dbloss 1 --des exp --gpu 0 --use_gpu 1 --seed 2025
