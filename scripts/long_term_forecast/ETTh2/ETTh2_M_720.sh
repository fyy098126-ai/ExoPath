#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ../Time-Series-Library/dataset/ETT-small/ --data_path ETTh2.csv --data ETTh2 --model ExoPath --model_id ETTh2_M_96_720 --features M --target OT --seq_len 96 --label_len 48 --pred_len 720 --enc_in 7 --dec_in 7 --c_out 7 --learning_rate 0.0015 --weight_decay 3e-08 --lradj cos_warmup --patience 5 --d_model 16 --t_ff 512 --num_groups 48 --batch_size 256 --dropout 0.175 --dbloss_alpha 0.05 --dbloss_beta 0.3 --train_epochs 30 --use_dbloss 1 --des exp --gpu 0 --use_gpu 1 --seed 2025
