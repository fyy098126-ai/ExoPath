#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ../Time-Series-Library/dataset/electricity/ --data_path electricity.csv --data custom --model ExoPath --model_id electricity_M_96_336 --features M --seq_len 96 --label_len 48 --pred_len 336 --enc_in 321 --dec_in 321 --c_out 321 --learning_rate 9.693576254163805e-05 --lradj 'type3' --patience 15 --d_model 768 --t_ff 6144 --num_groups 32 --batch_size 8 --dropout 0.30000000000000004 --dbloss_alpha 0.3 --dbloss_beta 0.7500000000000001 --weight_decay 0.0 --train_epochs 30 --use_dbloss 1 --des exp --gpu 0 --use_gpu 1 --seed 2021

