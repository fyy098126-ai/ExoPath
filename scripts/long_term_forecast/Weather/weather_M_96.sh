#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ../Time-Series-Library/dataset/weather/ --data_path weather.csv --data custom --model ExoPath --model_id weather_M_96_96 --features M --target OT --seq_len 96 --label_len 48 --pred_len 96 --enc_in 21 --dec_in 21 --c_out 21 --learning_rate 0.0002 --weight_decay 0.0 --lradj cos_warmup --patience 15 --d_model 64 --t_ff 5120 --num_groups 48 --batch_size 16 --dropout 0.0 --dbloss_alpha 0.25 --dbloss_beta 0.2 --train_epochs 30 --use_dbloss 1 --des exp --gpu 0 --use_gpu 1 --seed 2025
