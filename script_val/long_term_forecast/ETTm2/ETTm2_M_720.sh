#!/bin/bash
set -e

# Study: XGate_Omni_ETTm2_M_720_VAL
# 指定 Trial: Trial #1
# Best Validation MSE: 0.2808091342
# Test MSE @ Best Validation: 0.3865652382
# Test MAE @ Best Validation: 0.3863871694
# Best Validation Epoch: N/A

export CUDA_VISIBLE_DEVICES=1
cd /home/mlc8/fy/MySeriesModel

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ../Time-Series-Library/dataset/ETT-small/ --data_path ETTm2.csv --data ETTm2 --model ExoPath --model_id ETTm2_M_96_720 --features M --target OT --seq_len 96 --label_len 48 --pred_len 720 --enc_in 7 --dec_in 7 --c_out 7 --learning_rate 0.0002 --weight_decay 0.0 --lradj type3 --patience 5 --d_model 512 --t_ff 5120 --num_groups 2 --batch_size 64 --dropout 0.6 --dbloss_alpha 0.2 --dbloss_beta 0.65 --train_epochs 30 --use_dbloss 1 --des exp --gpu 0 --use_gpu 1 --seed 2025
