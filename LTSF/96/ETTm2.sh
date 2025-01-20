#!/bin/bash
mkdir -p ./logs/LongForecasting/ETTm2

export CUDA_VISIBLE_DEVICES=1
model_name=LiNo
seq_lens=(96)
bss=(128)
lrs=(1e-5)
log_dir="./logs/LongForecasting/ETTm2/"
layers=(1)
pred_lens=(96)
dropouts=(0.2)
betas=(0.5)
d_models=(512)

for bs in "${bss[@]}"; do
    for lr in "${lrs[@]}"; do
        for layer in "${layers[@]}"; do
            for dropout in "${dropouts[@]}"; do
                for beta in "${betas[@]}"; do
                    for d_model in "${d_models[@]}"; do
                        for pred_len in "${pred_lens[@]}"; do
                            for seq_len in "${seq_lens[@]}"; do
                                python -u run.py \
                                --task_name long_term_forecast \
                                --is_training 1 \
                                --root_path /data/gqyu/dataset/ETT-small/ \
                                --data_path ETTm2.csv \
                                --model_id "ETTm2_${seq_len}_${pred_len}" \
                                --model $model_name \
                                --data ETTm2 \
                                --initial 1\
                                --features M \
                                --seq_len $seq_len \
                                --pred_len $pred_len \
                                --batch_size $bs \
                                --learning_rate $lr \
                                --layers $layer\
                                --dropout $dropout\
                                --beta $beta\
                                --d_model $d_model\
                                --enc_in 7 \
                                --dec_in 7 \
                                --c_out 7 \
                                --train_epochs 40\
                                --des 'Exp' \
                                --itr 1 >"${log_dir}bs${bs}_lr${lr}_lay${layer}_dp${dropout}_beta${beta}_dm${d_model}_${pred_len}_${seq_len}.log"
                            done
                        done
                    done
                done
            done
        done
    done
done
