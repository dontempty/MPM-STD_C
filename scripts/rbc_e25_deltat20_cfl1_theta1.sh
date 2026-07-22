#!/bin/bash
#SBATCH -J rbc_e25_cfl1
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 16:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e25_deltat20_cfl1_theta1.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E25: DeltaT=20K, 256^3, COLD START, MaxCFL=1.0, theta=1.0 (no BLKRELAX -- momentum bug-1 + thermal bug-6/7 fixes applied) ==="
mpirun -np 2 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e25_deltat20_cfl1_theta1.toml 2>&1 \
  | grep --line-buffered -E "step=|restart|nan|NaN|error|Error|Nu\]"
echo "=== E25 done $(date) ==="
