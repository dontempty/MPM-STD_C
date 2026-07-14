#!/bin/bash
#SBATCH -J rbc_e10_relax
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 24:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e10_relax_prod.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E10: h-256 production continuation with block-coupling under-relaxation theta=0.5 ==="
echo "=== spike-free clean stats window: nstat_start=1755000 (t~14562), ~140k steps to t~15950 ==="
mkdir -p apps/rbc/e10_relax_prod
mpirun -np 2 -x MPMSTD_BLKRELAX=0.5 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e10_relax_prod.toml 2>&1 \
  | grep --line-buffered -E "step=|restart|nan|NaN|error|Error"
echo "=== E10 done ==="
