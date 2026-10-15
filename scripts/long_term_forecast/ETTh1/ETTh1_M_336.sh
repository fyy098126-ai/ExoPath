#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ../Time-Series-Library/dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 --model ExoPath --model_id ETTh1_M_96_336 --features M --seq_len 96 --label_len 48 --pred_len 336 --enc_in 7 --dec_in 7 --c_out 7 --learning_rate 0.0022392956565151323 --lradj 'cos_warmup' --patience 7 --d_model 64 --t_ff 0 --num_groups 6 --batch_size 8 --dropout 0.15000000000000002 --dbloss_alpha 0.8500000000000001 --dbloss_beta 0.8500000000000001 --weight_decay 0.0 --train_epochs 30 --use_dbloss 1 --des exp --gpu 0 --use_gpu 1 --seed 2024

