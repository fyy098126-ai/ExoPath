#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ../Time-Series-Library/dataset/traffic/ --data_path traffic.csv --data custom --model ExoPath  --model_id traffic_MS_96_96 --features MS --target OT --seq_len 96 --label_len 48 --pred_len 96 --enc_in 862 --dec_in 862 --c_out 1 --learning_rate 0.0003 --weight_decay 0.0 --lradj type3 --patience 15 --d_model 640 --t_ff 1024  --num_groups 16 --batch_size 16 --dropout 0.25 --dbloss_alpha 0.6 --dbloss_beta 0.1 --train_epochs 30 --use_dbloss 1 --des exp --gpu 0 --use_gpu 1 --seed 2025
