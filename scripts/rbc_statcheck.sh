#!/bin/bash
#SBATCH -J statcheck
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/statcheck.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
mpirun -np 1 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_statcheck.toml 2>&1 | tail -40
