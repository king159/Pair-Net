#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
torchrun --nnodes=2 --nproc_per_node=$GPUS --rdzv_backend=c10d --rdzv_endpoint=localhost:29411 --rdzv_id=0 $(dirname "$0")/train.py $CONFIG --launcher pytorch