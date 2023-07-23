#!/usr/bin/env bash
#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=q_ug2x24
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --job-name=q200
#SBATCH --output=cluster_output/%j_%x.out
#SBATCH --error=cluster_output/%j_%x.err
#SBATCH --nodelist=SCSEGPU-TC1-04
module load anaconda
source activate did
cd /home/FYP/c190209/C190209/PSG-mix
export OMP_NUM_THREADS=72
export MKL_NUM_THREADS=72
export PYTHONPATH='.':$PYTHONPATH
bash tools/dist_train.sh configs/mask2former/cross2_r50_q200_psg.py 2
# /home/FYP/c190209/.conda/envs/psg/bin/python tools/train.py configs/reformer/did_r50_psg.py