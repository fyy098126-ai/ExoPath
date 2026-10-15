#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ../Time-Series-Library/dataset/ETT-small/ --data_path ETTm2.csv --data ETTm2 --model ExoPath --model_id ETTm2_M_96_192 --features M --target OT --seq_len 96 --label_len 48 --pred_len 192 --enc_in 7 --dec_in 7 --c_out 7 --learning_rate 0.0005 --weight_decay 1e-08 --lradj cosine --patience 5 --d_model 640 --t_ff 256 --num_groups 24 --batch_size 8 --dropout 0.75 --dbloss_alpha 0.15 --dbloss_beta 0.6 --train_epochs 30 --use_dbloss 1 --des exp --gpu 0 --use_gpu 1 --seed 2025
