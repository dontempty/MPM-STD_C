#!/bin/bash
#SBATCH -J rbc_e9d_blk2
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 01:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e9d_blk2.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E9d: block-coupling no-op only at top 2 wall-adjacent rows (MPMSTD_BLKMASK_TOP=2), 600 steps ==="
mkdir -p apps/rbc/e9d_blk2
mpirun -np 2 -x MPMSTD_BLKMASK_TOP=2 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e9d_blk2.toml 2>&1 \
  | grep --line-buffered -E "step=|restart|nan|NaN|error|Error"
echo "=== E9d done ==="
