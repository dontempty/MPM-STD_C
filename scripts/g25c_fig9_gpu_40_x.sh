#!/bin/bash
#SBATCH -J fig9_40x
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 04:30:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/g25c_fig9_gpu_40_x.log

cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7

echo "=== G25c: Fig9 GPU ΔT=40K EXTENDED (350K steps, ~3h) ==="
echo "nstat_start=50K (early flow); 300K step average → ~2400 FF ≈ 3.5×τ"
echo ""

mpirun -np 2 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_fig9_gpu_nob_40_x.toml 2>&1 \
  | grep --line-buffered -E "step=|Nu\]|TIMING|TIMING_SS|restart"
