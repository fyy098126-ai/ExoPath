#!/bin/bash
set -e


export CUDA_VISIBLE_DEVICES=1

python -u run.py  --task_name long_term_forecast  --is_training 1  --root_path ../Time-Series-Library/dataset/ETT-small/  --data_path ETTh1.csv  --data ETTh1 --model ExoPath   --model_id ETTh1_M_96_720   --features M  --seq_len 96  --label_len 48  --pred_len 720  --enc_in 7  --dec_in 7  --c_out 7  --learning_rate 0.0013760413464609872  --weight_decay 3.0976471037790353e-06 --lradj 'type1'  --patience 3  --d_model 512  --t_ff 4096   --num_groups 24  --batch_size 96  --dropout 0.55  --dbloss_alpha 0.5  --dbloss_beta 0.8500000000000001  --train_epochs 30  --use_dbloss 1  --des exp  --gpu 0  --use_gpu 1  --seed 2021

