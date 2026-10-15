#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python -u run.py --task_name long_term_forecast   --is_training 1   --root_path ../Time-Series-Library/dataset/EPF/  --data_path BE.csv   --data custom  --model ExoPath  --model_id BE_MS_168_24   --features MS  --seq_len 168   --label_len 84   --pred_len 24   --enc_in 3   --dec_in 3   --c_out 1   --learning_rate 0.0006368133933086496   --weight_decay 2.620800401732645e-07   --lradj 'cosine'  --patience 20  --d_model 512  --t_ff 1536   --num_groups 8  --batch_size 8  --dropout 0.5  --dbloss_alpha 0.2 --dbloss_beta 0.3  --train_epochs 30  --use_dbloss 1  --des exp  --gpu 0  --use_gpu 1  --seed 2021

