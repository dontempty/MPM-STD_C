#!/bin/bash
#SBATCH -J rbc_runc_cfl1
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 12:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_runc_e22_cfl1.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== Run C: E22 restart + NEW binary (D5/D2 fixed), CFL=1.0 (only change vs Run B), np=1 ==="
echo "=== discriminator: does the CFL=1.0 residual near-wall mode pull theta_c 0.128 -> ~0.122? ==="
mpirun -np 1 -x MPMSTD_BLKRELAX=0.5 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_runc_e22_cfl1.toml 2>&1 \
  | tee apps/rbc/logs/rbc_runc_e22_cfl1_full.log | grep --line-buffered -E "restart|nan|NaN|error|Error"
echo "=== Run C done $(date) ==="
