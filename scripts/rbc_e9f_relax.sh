#!/bin/bash
#SBATCH -J rbc_e9f_relax
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 01:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e9f_relax.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E9f: uniform block-coupling under-relaxation theta=0.5 (no spatial edge), 600 steps ==="
make BACKEND=cuda apps 2>&1 | tail -3
mkdir -p apps/rbc/e9f_relax
mpirun -np 2 -x MPMSTD_BLKRELAX=0.5 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e9f_relax.toml 2>&1 \
  | grep --line-buffered -E "step=|restart|nan|NaN|error|Error"
echo "=== E9f done ==="
