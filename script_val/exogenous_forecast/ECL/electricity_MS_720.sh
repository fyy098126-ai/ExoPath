#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1


python -u run.py --task_name long_term_forecast --is_training 1 --root_path ../Time-Series-Library/dataset/electricity/ --data_path electricity.csv --data custom --model ExoPath --model_id electricity_MS_96_720 --features MS --target OT --seq_len 96 --label_len 48 --pred_len 720 --enc_in 321 --dec_in 321 --c_out 1 --learning_rate 0.0007 --weight_decay 0.0 --lradj cosine --patience 5 --d_model 640 --t_ff 3072 --num_groups 64 --batch_size 16 --dropout 0.75 --dbloss_alpha 0.45 --dbloss_beta 0.05 --train_epochs 30 --use_dbloss 1 --des exp --gpu 0 --use_gpu 1 --seed 2025
