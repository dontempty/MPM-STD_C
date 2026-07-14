#!/bin/bash
#SBATCH -J rbc_e14_clipfix
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 08:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e14_clipfix.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E14: T-clip ghost bug FIXED (interior-only clip) + theta=0.25 dt0.02, statzero window ==="
echo "=== expected: wall-gap flux kink gone; theta_c moves from 0.066 toward 0.071 ==="
make BACKEND=cuda apps 2>&1 | tail -3
mkdir -p apps/rbc/e14_clipfix
mpirun -np 2 -x MPMSTD_BLKRELAX=0.25 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e14_clipfix.toml 2>&1 \
  | grep --line-buffered -E "step=|restart|nan|NaN|error|Error"
echo "=== E14 done ==="
