#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ../Time-Series-Library/dataset/ETT-small/ --data_path ETTm1.csv --data ETTm1 --model ExoPath --model_id ETTm1_MS_96_192 --features MS --target OT --seq_len 96 --label_len 48 --pred_len 192 --enc_in 7 --dec_in 7 --c_out 1 --learning_rate 0.05 --weight_decay 0.0 --lradj type3 --patience 10 --d_model 512 --t_ff 5120 --num_groups 4 --batch_size 128 --dropout 0.25 --dbloss_alpha 0.45 --dbloss_beta 0.25 --train_epochs 30 --use_dbloss 1 --des exp --gpu 0 --use_gpu 1 --seed 2025
