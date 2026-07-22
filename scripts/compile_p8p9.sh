#!/bin/bash
#SBATCH -J compile_p8p9
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/compile_p8p9.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
make BACKEND=cuda apps 2>&1 | tail -6
ls -la build/cuda/bin/apps/rbc_gpu_nob
