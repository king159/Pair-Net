#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
torchrun --nnodes=1 --nproc_per_node=$GPUS --rdzv_backend=c10d --rdzv_endpoint=localhost:0 $(dirname "$0")/train.py $CONFIG --launcher pytorch