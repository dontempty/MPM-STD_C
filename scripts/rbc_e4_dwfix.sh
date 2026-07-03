#!/bin/bash
#SBATCH -J rbc_e4_dwfix
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 04:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e4_dwfix.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E4: dW lower-wall-face pin fix, 128^3 warm start from t=7185 spiky state ==="
echo "=== expect: k=0 U spike decays; theta_c drifts UP from 0.046 if fix is causal ==="
mkdir -p apps/rbc/e4_dwfix_warm
mpirun -np 2 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e4_dwfix_warm.toml 2>&1 | grep --line-buffered -E "restart|nan|NaN|error|Error"
echo "=== E4 done ==="
