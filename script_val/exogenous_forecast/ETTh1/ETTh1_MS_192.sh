#!/bin/bash
set -e


export CUDA_VISIBLE_DEVICES=0

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ../Time-Series-Library/dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 --model ExoPath --model_id ETTh1_MS_96_192 --features MS --target OT --seq_len 96 --label_len 48 --pred_len 192 --enc_in 7 --dec_in 7 --c_out 1 --learning_rate 0.007 --weight_decay 0.0 --lradj cos_warmup --patience 15 --d_model 64 --t_ff 3072 --num_groups 6 --batch_size 16 --dropout 0.1 --dbloss_alpha 0.3 --dbloss_beta 0.5 --train_epochs 30 --use_dbloss 1 --des exp --gpu 0 --use_gpu 1 --seed 2025
