#!/bin/bash
#SBATCH -J rbc_e9c_regrow
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 01:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e9c_regrow.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E9c: probes ON + 16-plane block mask — who pumps during regrowth? ==="
make BACKEND=cuda apps 2>&1 | tail -3
mkdir -p apps/rbc/e9c_regrow
mpirun -np 2 -x MPMSTD_MOMDBG=1 -x MPMSTD_BLKMASK_TOP=16 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e9c_regrow.toml
echo "=== E9c done ==="
