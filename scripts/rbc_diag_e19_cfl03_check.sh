#!/bin/bash
#SBATCH -J rbc_diag_c03
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_diag_e19_cfl03_check.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== CHECK: restart_e19 + 1000 steps, MaxCFL=0.3, dt_start=0.05 (UNCAPPED), NEW (bug-fixed) binary -- does dt match old E19's ~0.007-0.009, or differ? ==="
mpirun -np 1 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_diag_e19_cfl03_check.toml 2>&1 \
  | grep --line-buffered -E "step=|restart|nan|NaN|error|Error"
echo "=== CHECK done $(date) ==="
