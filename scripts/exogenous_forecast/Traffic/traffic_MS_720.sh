#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ../Time-Series-Library/dataset/traffic/ --data_path traffic.csv --data custom --model ExoPath --model_id traffic_MS_96_720 --features MS --target OT --seq_len 96 --label_len 48 --pred_len 720 --enc_in 862 --dec_in 862 --c_out 1 --learning_rate 0.003 --weight_decay 3e-06 --lradj type3 --patience 7 --d_model 448 --t_ff 256 --num_groups 32 --batch_size 48 --dropout 0.45 --dbloss_alpha 0.5 --dbloss_beta 0.15 --train_epochs 30 --use_dbloss 1 --des exp --gpu 0 --use_gpu 1 --seed 2025
