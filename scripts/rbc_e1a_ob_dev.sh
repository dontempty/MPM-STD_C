#!/bin/bash
#SBATCH -J rbc_e1a_ob_dev
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 04:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e1a_ob_dev.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E1a: OB development run (DeltaT=0.001, 125K steps, writes restart_e1_ob) ==="
mkdir -p apps/rbc/e1_ob_dev
mpirun -np 2 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e1a_ob_dev.toml 2>&1 \
  | grep --line-buffered -E "step=|Nu|restart|nan|NaN|error" | awk 'NR%20==1 || /restart|nan|error/'
echo "=== E1a done ==="
