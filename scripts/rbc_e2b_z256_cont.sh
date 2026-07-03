#!/bin/bash
#SBATCH -J rbc_e2b_z256_cont
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 24:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e2b_z256_cont.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E2: E2 continuation t=8000-12000, plateau detection ==="
echo "=== tests whether discrete BL resolution causes the sqrt(R) contrast loss ==="
mkdir -p apps/rbc/e2b_z256_cont
mpirun -np 2 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e2b_z256_cont.toml 2>&1 \
  | grep --line-buffered -E "step=|Nu|restart|nan|NaN|error" | awk 'NR%20==1 || /restart|nan|error/'
echo "=== E2 done ==="
