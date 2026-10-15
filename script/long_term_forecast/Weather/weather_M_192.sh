#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ../Time-Series-Library/dataset/weather/ --data_path weather.csv --data custom --model ExoPath --model_id weather_M_96_192 --features M --target OT --seq_len 96 --label_len 48 --pred_len 192 --enc_in 21 --dec_in 21 --c_out 21 --learning_rate 0.0005 --weight_decay 1e-07 --lradj cosine --patience 15 --d_model 320 --t_ff 1536 --num_groups 24 --batch_size 16 --dropout 0.7 --dbloss_alpha 0.25 --dbloss_beta 0.1 --train_epochs 30 --use_dbloss 1 --des exp --gpu 0 --use_gpu 1 --seed 2025
