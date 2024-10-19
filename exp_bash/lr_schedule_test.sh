#!/bin/bash

# 学习率调度实验
EXPERIMENT_NAME="learning_rate_schedule"
EXPERIMENT_DIR="./experiment/$EXPERIMENT_NAME"
DATA_DIR="./data"
BATCH_SIZE=128
EPOCHS=300
LR=0.05
WEIGHT_DECAY=0.0

# 创建实验目录
mkdir -p $EXPERIMENT_DIR

# 恒定学习率和余弦退火调度的实验
for SCHEDULER in "constant" "cosine"
do
    OUTPUT_FILE="scheduler=${SCHEDULER}"
    USE_SCHEDULER_FLAG=""

    if [ "$SCHEDULER" == "cosine" ]; then
        USE_SCHEDULER_FLAG="--use-scheduler"
    fi

    python ../main.py --data-dir $DATA_DIR --batch-size $BATCH_SIZE --epochs $EPOCHS \
    --lr $LR --weight-decay $WEIGHT_DECAY --augment $USE_SCHEDULER_FLAG \
    --output-dir $EXPERIMENT_DIR --output-file $OUTPUT_FILE
done
