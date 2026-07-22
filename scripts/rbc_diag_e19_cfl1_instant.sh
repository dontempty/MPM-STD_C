#!/bin/bash
#SBATCH -J rbc_diag_e19c1
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 02:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_diag_e19_cfl1_instant.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== DIAG: restart_e19 (equilibrated DeltaT=20K, +0.0647, step=2380000) + 20000 steps, MaxCFL=1.0, theta=1.0 (no BLKRELAX, bug-fixed code), instant Theta_c/Nu every 100 steps -- does it stay near +0.0647 or drift negative like E25? ==="
mpirun -np 1 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_diag_e19_cfl1_instant.toml 2>&1 \
  | grep --line-buffered -E "step=|restart|nan|NaN|error|Error|Nu\]"
echo "=== DIAG done $(date) ==="
