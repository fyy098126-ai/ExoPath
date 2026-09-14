#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ../Time-Series-Library/dataset/electricity/ --data_path electricity.csv --data custom --model ExoPath --model_id electricity_MS_96_336 --features MS --target OT --seq_len 96 --label_len 48 --pred_len 336 --enc_in 321 --dec_in 321 --c_out 1 --learning_rate 0.007 --weight_decay 0.0 --lradj type1 --patience 15 --d_model 192 --t_ff 5120 --num_groups 64 --batch_size 64 --dropout 0.15 --dbloss_alpha 0.35 --dbloss_beta 0.35 --train_epochs 30 --use_dbloss 1 --des exp --gpu 0 --use_gpu 1 --seed 2025
