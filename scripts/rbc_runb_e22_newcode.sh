#!/bin/bash
#SBATCH -J rbc_runb_e22new
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 12:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_runb_e22_newcode.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== Run B: E22 restart (theta_c=0.128, t=8644) + NEW binary, settings matched to E22 (CFL=0.3, theta=0.5) ==="
echo "=== discriminator: does instant theta_c decay toward the new-code attractor (~0.1226)? ==="
mpirun -np 1 -x MPMSTD_BLKRELAX=0.5 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_runb_e22_newcode.toml 2>&1 \
  | tee apps/rbc/logs/rbc_runb_e22_newcode_full.log | grep --line-buffered -E "restart|nan|NaN|error|Error"
echo "=== Run B done $(date) ==="
