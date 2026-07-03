#!/bin/bash
#SBATCH -J rbc_e5_z512
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 12:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e5_z512.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E5: 128x128x512 (fixed binary), warm start from z-interpolated E2 field (t=8000) ==="
echo "=== decisive: theta_c ~0.063 -> resolution story matches ref 0.0744; ~0.057 -> repo!=production ==="
mkdir -p apps/rbc/e5_z512
mpirun -np 2 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e5_z512.toml 2>&1 | grep --line-buffered -E "restart|nan|NaN|error|Error"
echo "=== E5 done ==="
