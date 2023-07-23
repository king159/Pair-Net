#!/bin/bash
# sh scripts/psgformer/psgformer_test.sh

GPU=1
CPU=4
node=67
PORT=29500
jobname=pairnet

PYTHONPATH='.':$PYTHONPATH \
python tools/test.py \
     configs/psgformer/mask2former_play.py \
     work_dirs/checkpoints/mask2former_r50_lsj_8x2_50e_coco-panoptic_20220326_224516-11a44721.pth \
     --out work_dirs/mask2former_play/result.pkl \
     --eval PQ
