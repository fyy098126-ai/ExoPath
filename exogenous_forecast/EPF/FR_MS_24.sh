#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ../Time-Series-Library/dataset/EPF/ --data_path FR.csv --data custom --model ExoPath --model_id FR_MS_168_24 --features MS --target OT --seq_len 168 --label_len 84 --pred_len 24 --enc_in 3 --dec_in 3 --c_out 1 --learning_rate 0.0003 --weight_decay 0.0 --lradj cos_warmup --patience 10 --d_model 640 --t_ff 4096 --num_groups 12 --batch_size 8 --dropout 0.5 --dbloss_alpha 0.05 --dbloss_beta 0.65 --train_epochs 30 --use_dbloss 1 --des exp --gpu 0 --use_gpu 1 --seed 2025
