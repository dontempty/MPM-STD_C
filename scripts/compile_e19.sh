#!/bin/bash
#SBATCH -J compile_e19
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:15:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/compile_e19.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
make BACKEND=cuda apps 2>&1 | tail -20
ls -la build/cuda/bin/apps/rbc_gpu_nob
