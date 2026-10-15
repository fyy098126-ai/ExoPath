#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ../Time-Series-Library/dataset/traffic/ --data_path traffic.csv --data custom --model ExoPath --model_id traffic_M_96_720 --features M --target OT --seq_len 96 --label_len 48 --pred_len 720 --enc_in 862 --dec_in 862 --c_out 862 --learning_rate 0.001 --weight_decay 0.0 --lradj cosine --patience 15 --d_model 384 --t_ff 256 --num_groups 96 --batch_size 48 --dropout 0.35 --dbloss_alpha 0.15 --dbloss_beta 0.3 --train_epochs 30 --use_dbloss 1 --des exp --gpu 0 --use_gpu 1 --seed 2025
