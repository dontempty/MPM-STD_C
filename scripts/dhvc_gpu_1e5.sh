#!/bin/bash
#SBATCH -J dhvc_1e5
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 06:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/dhvc_gpu_1e5.log

cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7

echo "=== DHVC GPU Ra=1e5 LONG (300K steps ~t700 ~18 periods, nstat_start=15K, nstat_out=20K, 2xA100 gpu01) ==="
mpirun -np 2 build/cuda/bin/apps/dhvc_gpu apps/dhvc/input_fig7_gpu_1e5.toml 2>&1 \
  | grep --line-buffered -E "step=|dhvc_gpu\]|stats out|rror|Killed|CANCELLED|BLOW|nan"
