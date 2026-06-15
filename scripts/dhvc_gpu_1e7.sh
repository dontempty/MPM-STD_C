#!/bin/bash
#SBATCH -J dhvc_1e7
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 06:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/dhvc_gpu_1e7.log

cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7

echo "=== DHVC GPU Ra=1e7 (30K steps, 512x128x256, 2xA100 gpu02) ==="
mpirun -np 2 build/cuda/bin/apps/dhvc_gpu apps/dhvc/input_fig7_gpu_1e7.toml 2>&1 \
  | grep --line-buffered -E "step=|Re_dstar|dhvc_gpu\]|error:|Error:|Killed"
