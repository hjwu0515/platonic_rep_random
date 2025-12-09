#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=100G
#SBATCH -t 06:00:00
#SBATCH -J comp_layer
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err


set -euo pipefail

module load miniforge/24.3.0-0
conda activate platrep

python compute_best_layer.py \
  --dataset minhuh/prh \
  --subset wit_1024 \
  --modelset val \
  --modality_x language \
  --pool_x avg \
  --modality_y vision \
  --pool_y cls
