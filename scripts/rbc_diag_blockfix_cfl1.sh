#!/bin/bash
#SBATCH -J rbc_diag_blockfix
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 01:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_diag_blockfix_cfl1.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== DIAG blockfix: restart_e19 (equilibrated, step=2380000) + 2000 steps, MaxCFL=1.0, theta=1.0, MPMSTD_CFLDBG=1 ==="
echo "=== rearranged block coupling (MPM-STD_main arrangement): wall-normal dW solved last, no post-hoc dW correction ==="
mpirun -np 1 -x MPMSTD_CFLDBG=1 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_diag_blockfix_cfl1.toml 2>&1 \
  | tee apps/rbc/logs/rbc_diag_blockfix_cfl1_full.log \
  | grep --line-buffered -E "restart|nan|NaN|error|Error" 
echo "=== DIAG done $(date) ==="
