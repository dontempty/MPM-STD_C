#!/bin/bash
#SBATCH -J lean_diag
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:10:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/dhvc_lean_diag.log

# Diagnose the LEAN crash: run the lean binary with FULL output (no grep filter)
# on a short cold-start config → capture the exact CUDA error / where it dies.
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== lean binary, 20 steps, full output ==="
CUDA_LAUNCH_BLOCKING=1 mpirun -np 2 build/cuda/bin/apps/dhvc_gpu_lean apps/dhvc/input_prof.toml 2>&1 | tail -40
echo "=== exit: $? ==="
