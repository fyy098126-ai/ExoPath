#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ../Time-Series-Library/dataset/electricity/ --data_path electricity.csv --data custom --model ExoPath --model_id electricity_M_96_96 --features M --seq_len 96 --label_len 48 --pred_len 96 --enc_in 321 --dec_in 321 --c_out 321 --learning_rate 0.0039584612100526904 --weight_decay 1.3587991900751026e-07 --lradj 'cosine' --patience 3 --d_model 384 --t_ff 3072 --num_groups 32 --batch_size 64 --dropout 0.15000000000000002 --dbloss_alpha 0.15000000000000002 --dbloss_beta 0.55 --train_epochs 30 --use_dbloss 1 --des exp --gpu 0 --use_gpu 1 --seed 2021
