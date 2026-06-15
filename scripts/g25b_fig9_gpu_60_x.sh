#!/bin/bash
#SBATCH -J fig9_60x
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 08:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/g25b_fig9_gpu_60_x.log

cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7

echo "=== G25b: Fig9 GPU ΔT=60K EXTENDED (400K steps, ~7.5h) ==="
echo "nstat_start=100K (t≈500FF≈0.73×τ); 300K step average → ~1500 FF ≈ 2.2×τ"
echo ""

mpirun -np 2 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_fig9_gpu_nob_60_x.toml 2>&1 \
  | grep --line-buffered -E "step=|Nu\]|TIMING|TIMING_SS|restart"
