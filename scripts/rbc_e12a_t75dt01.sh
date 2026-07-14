#!/bin/bash
#SBATCH -J rbc_e12a_t75dt01
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 03:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e12a_t75dt01.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E12a: theta=0.75, dt cap 0.01 — product 0.0075, stability margin probe ==="
echo "=== watch: dt level reached, umax/top-slab probe stability, div ==="
mkdir -p apps/rbc/e12a_t75dt01
mpirun -np 2 -x MPMSTD_BLKRELAX=0.75 -x MPMSTD_MOMDBG=1 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e12a_t75dt01.toml 2>&1 \
  | grep --line-buffered -E "step=|postmom|postproj|restart|nan|NaN|error|Error"
echo "=== E11 done ==="
