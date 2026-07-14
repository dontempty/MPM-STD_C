#!/bin/bash
#SBATCH -J rbc_e8b_lowcfl
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 03:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e8b_lowcfl.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E8 A/B: CFL=0.15 A-B: does the 2dx wall-row mode have a dt threshold? ==="
echo "=== verdict: umax decay => semi-implicit convection dt threshold at wall row ==="
mkdir -p apps/rbc/e8b_lowcfl
mpirun -np 2 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e8b_lowcfl.toml 2>&1 \
  | grep --line-buffered -E "step=|restart|nan|NaN|error|Error"
echo "=== E8 done ==="
