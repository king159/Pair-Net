#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
MODE=$3
GPUS=$4
torchrun --nnodes=1 --nproc_per_node=$GPUS --rdzv_backend=c10d --rdzv_endpoint=localhost:0 $(dirname "$0")/test.py $CONFIG $CHECKPOINT --eval $MODE --launcher pytorch