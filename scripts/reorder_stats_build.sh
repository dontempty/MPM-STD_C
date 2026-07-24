#!/bin/bash
#SBATCH -J reo_stats_build
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:25:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/reorder_stats_build.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
rm -f build/cuda/bin/apps/rbc_gpu_nob
make BACKEND=cuda apps 2>&1 | tail -25
echo "=== binary ==="; ls -la build/cuda/bin/apps/rbc_gpu_nob 2>&1
