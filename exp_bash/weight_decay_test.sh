#!/bin/bash

# 权重衰减实验
EXPERIMENT_NAME="weight_decay"
EXPERIMENT_DIR="./experiment/$EXPERIMENT_NAME"
DATA_DIR="./data"
BATCH_SIZE=128
EPOCHS=300
LR=0.05

# 创建实验目录
mkdir -p $EXPERIMENT_DIR

# 不同权重衰减系数的实验
for WEIGHT_DECAY in 5e-4 1e-4
do
    OUTPUT_FILE="weight_decay=${WEIGHT_DECAY}"
    MODEL_NAME="WD=${WEIGHT_DECAY}"

    python ../main.py --data-dir $DATA_DIR --batch-size $BATCH_SIZE --epochs $EPOCHS \
    --lr $LR --weight-decay $WEIGHT_DECAY --augment --use-scheduler --test --save-model \
    --name-model "$MODEL_NAME" --output-dir $EXPERIMENT_DIR --output-file $OUTPUT_FILE
done
