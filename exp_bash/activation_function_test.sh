#!/bin/bash

# 激活函数实验
EXPERIMENT_NAME="activation_function"
EXPERIMENT_DIR="./experiment/$EXPERIMENT_NAME"
DATA_DIR="./data"
BATCH_SIZE=128
EPOCHS=300
LR=0.05
WEIGHT_DECAY=5e-4

# 创建实验目录
mkdir -p $EXPERIMENT_DIR

# 使用 Sigmoid 激活函数的实验
OUTPUT_FILE="activation=sigmoid"
SIGMOID_BLOCK_INDICES="4 5 6 7 8 9 10"

python ../main.py --data-dir $DATA_DIR --batch-size $BATCH_SIZE --epochs $EPOCHS --test \
--lr $LR --weight-decay $WEIGHT_DECAY --augment --use-scheduler --save-l2-norm --save-model \
--sigmoid-block-ind $SIGMOID_BLOCK_INDICES --output-dir $EXPERIMENT_DIR --output-file $OUTPUT_FILE
