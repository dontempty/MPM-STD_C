#!/bin/bash
#SBATCH -J rbc_e1b_nob_warm
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 06:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e1b_nob_warm.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E1b: NOB DeltaT=20K warm-started from OB field (Kim-protocol test) ==="
echo "=== compare theta_c(t) vs cold-start plateau 0.046 ==="
mkdir -p apps/rbc/e1_nob_warm
mpirun -np 2 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e1b_nob_warm.toml 2>&1 \
  | grep --line-buffered -E "step=|Nu|restart|nan|NaN|error" | awk 'NR%20==1 || /restart|nan|error/'
echo "=== E1b done ==="
