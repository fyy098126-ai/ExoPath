#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ../Time-Series-Library/dataset/ETT-small/ --data_path ETTh2.csv --data ETTh2 --model ExoPath --model_id ETTh2_MS_96_720 --features MS --target OT --seq_len 96 --label_len 48 --pred_len 720 --enc_in 7 --dec_in 7 --c_out 1 --learning_rate 0.0005 --weight_decay 0.0 --lradj type3 --patience 10 --d_model 192 --t_ff 1024 --num_groups 8 --batch_size 8 --dropout 0.15 --dbloss_alpha 0.15 --dbloss_beta 0.5 --train_epochs 30 --use_dbloss 1 --des exp --gpu 0 --use_gpu 1 --seed 2025
