#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ../Time-Series-Library/dataset/ETT-small/ --data_path ETTh2.csv --data ETTh2 --model ExoPath --model_id ETTh2_M_96_96 --features M --target OT --seq_len 96 --label_len 48 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --learning_rate 0.02 --weight_decay 3e-07 --lradj cos_warmup --patience 15 --d_model 640 --t_ff 5120 --num_groups 6 --batch_size 96 --dropout 0.15 --dbloss_alpha 0.35 --dbloss_beta 0.5 --train_epochs 30 --use_dbloss 1 --des exp --gpu 0 --use_gpu 1 --seed 2025
