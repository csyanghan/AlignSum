#!/bin/bash

# bart-large
MODEL_NAME_OR_PATH=./logs/checkpoints/cnn_dailymail/bart-large-cnn
TRAIN_DATA_PATH=data/cnn_dailymail/train_random1.jsonl
SAVE_NAME=bart-large-cnn-plus-human
PORT=8668
# SAVE_STEPS=400
# LOGGING_STEPS=40
# EPOCHS=3
SAVE_STEPS=10
LOGGING_STEPS=10
EPOCHS=20

rm -rf ./logs/checkpoints/cnn_dailymail/$SAVE_NAME

CUDA_VISIBLE_DEVICES="2" NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" torchrun --nproc_per_node 1 --master_port $PORT src/bart_ft.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --do_train \
    --do_eval False \
    --do_predict \
    --train_file $TRAIN_DATA_PATH \
    --test_file data/cnn_dailymail/test_random1.jsonl \
    --learning_rate 5e-5 \
    --num_train_epochs $EPOCHS \
    --text_column article \
    --summary_column pseudo_summary \
    --max_source_length 1024 \
    --max_target_length 128 \
    --output_dir ./logs/checkpoints/cnn_dailymail/$SAVE_NAME \
    --save_strategy steps \
    --save_steps  $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS \
    --save_total_limit 2 \
    --report_to wandb \
    --run_name cnn_dailymail_$SAVE_NAME  \
    --gradient_accumulation_steps 32 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 32 \
    --predict_with_generate