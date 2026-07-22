#!/bin/bash
#SBATCH -J reorder_build
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/reorder_clean_build.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== clean ==="
make BACKEND=cuda clean
echo "=== build BACKEND=cuda ==="
make BACKEND=cuda apps 2>&1 | tail -30
echo "=== binary check ==="
ls -la build/cuda/bin/apps/rbc_gpu_nob 2>&1
