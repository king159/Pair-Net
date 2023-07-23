#!/usr/bin/env bash
#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --job-name=pixel_input
#SBATCH --output=cluster_output/%x_%j.out
#SBATCH --error=cluster_output/%x_%j.err
#SBATCH --nodelist=SCSEGPU-TC1-01
module load anaconda
source activate did
export OMP_NUM_THREADS=72
export MKL_NUM_THREADS=72
export PYTHONPATH='.':$PYTHONPATH
/home/FYP/c190209/.conda/envs/did/bin/python tools/test.py /home/FYP/c190209/C190209/PSG-mix/configs/mask2former/cross2_r50_q200_psg.py /home/FYP/c190209/C190209/PSG-mix/work_dirs/psg_our_q200/epoch_4.pth --eval sgdet