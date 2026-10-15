#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ../Time-Series-Library/dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 --model ExoPath --model_id ETTh1_MS_96_720 --features MS --target OT --seq_len 96 --label_len 48 --pred_len 720 --enc_in 7 --dec_in 7 --c_out 1 --learning_rate 0.01 --weight_decay 1e-06 --lradj cosine --patience 15 --d_model 512 --t_ff 4096 --num_groups 6 --batch_size 32 --dropout 0.8 --dbloss_alpha 0.2 --dbloss_beta 0.45 --train_epochs 30 --use_dbloss 1 --des exp --gpu 0 --use_gpu 1 --seed 2025
