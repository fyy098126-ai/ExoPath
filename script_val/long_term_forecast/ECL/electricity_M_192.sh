#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ../Time-Series-Library/dataset/electricity/ --data_path electricity.csv --data custom --model ExoPath --model_id electricity_M_96_192 --features M --target OT --seq_len 96 --label_len 48 --pred_len 192 --enc_in 321 --dec_in 321 --c_out 321 --learning_rate 0.0005 --weight_decay 3e-07 --lradj type3 --patience 10 --d_model 256 --t_ff 5120 --num_groups 256 --batch_size 8 --dropout 0.1 --dbloss_alpha 0.3 --dbloss_beta 0.4 --train_epochs 30 --use_dbloss 1 --des exp --gpu 0 --use_gpu 1 --seed 2025
