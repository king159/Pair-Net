#!/bin/bash
# sh scripts/gpsnet/train_panoptic_fpn_r101_sgdet.sh

GPU=1
CPU=1
node=73
PORT=29500
jobname=pairnet

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python tools/train.py \
  configs/gpsnet/panoptic_fpn_r101_fpn_1x_sgdet_psg.py
