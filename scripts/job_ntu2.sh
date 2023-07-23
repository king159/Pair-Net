#!/usr/bin/env bash
#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=q_ug2x24
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --job-name=rellate
#SBATCH --output=cluster_output/%j_%x.out
#SBATCH --error=cluster_output/%j_%x.err
#SBATCH --nodelist=SCSEGPU-TC1-04
module load anaconda
source activate did
cd /home/FYP/ywang084/C190209/PSG-mix
export OMP_NUM_THREADS=72
export MKL_NUM_THREADS=72
export PYTHONPATH='.':$PYTHONPATH
/home/FYP/c190209/.conda/envs/did/bin/python tools/train.py configs/mask2former/cross2_r50_psg.py
# /home/FYP/c190209/.conda/envs/did/bin/python tools/test.py configs/mask2former/cross_r50_psg.py work_dirs/v100_clip01/epoch_1.pth --eval sgdet
