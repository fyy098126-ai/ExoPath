#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ../Time-Series-Library/dataset/ETT-small/ --data_path ETTm1.csv --data ETTm1 --model ExoPath --model_id ETTm1_M_96_720 --features M --target OT --seq_len 96 --label_len 48 --pred_len 720 --enc_in 7 --dec_in 7 --c_out 7 --learning_rate 0.003 --weight_decay 1e-07 --lradj cos_warmup --patience 7 --d_model 512 --t_ff 5120 --num_groups 12 --batch_size 8 --dropout 0.15 --dbloss_alpha 0.3 --dbloss_beta 0.7 --train_epochs 30 --use_dbloss 1 --des exp --gpu 0 --use_gpu 1 --seed 2025
