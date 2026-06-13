#!/bin/bash
#SBATCH -J fig9_20x
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 05:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/g25a_fig9_gpu_20_x.log

cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7

echo "=== G25a: Fig9 GPU ΔT=20K EXTENDED (500K steps, ~4.1h) ==="
echo "nstat_start=50K (t≈700FF≈1×τ); 450K step average → ~6300 FF ≈ 9×τ"
echo ""

mpirun -np 2 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_fig9_gpu_nob_20_x.toml 2>&1 \
  | grep --line-buffered -E "step=|Nu\]|TIMING|TIMING_SS|restart"
