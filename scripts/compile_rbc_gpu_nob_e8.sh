#!/bin/bash
#SBATCH -J compile_rbc_gpu_nob
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:20:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/compile_rbc_gpu_nob.log

cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7

echo "=== Compiling rbc_gpu_nob + dhvc_gpu ==="
make BACKEND=cuda apps 2>&1 | tail -20
echo "=== compile done ==="
ls -la build/cuda/bin/apps/rbc_gpu_nob build/cuda/bin/apps/dhvc_gpu 2>&1
