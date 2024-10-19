#!/bin/bash

# 学习率实验
EXPERIMENT_NAME="learning_rate"
EXPERIMENT_DIR="./experiment/$EXPERIMENT_NAME"
DATA_DIR="./data"
BATCH_SIZE=128
EPOCHS=15
WEIGHT_DECAY=0.0

# 创建实验目录
mkdir -p $EXPERIMENT_DIR

# 不同学习率的实验
for LR in 0.2 0.05 0.01
do
    OUTPUT_FILE="lr=${LR}"

    python ../main.py --data-dir $DATA_DIR --batch-size $BATCH_SIZE --epochs $EPOCHS \
    --lr $LR --weight-decay $WEIGHT_DECAY --augment --output-dir $EXPERIMENT_DIR --output-file $OUTPUT_FILE
done
