#!/bin/bash
#SBATCH -J rbc_e19_stats_reset
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 03:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e19_stats_reset.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E19: stats_reset=1, +20000 steps from restart_e18_buoyfix_h256 (t=20032) ==="
stdbuf -oL -eL mpirun -np 1 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e19_stats_reset.toml 2>&1 \
  | grep --line-buffered -E "step=|restart|nan|NaN|error|Error|Nu\]"
echo "=== E19 done $(date) ==="
