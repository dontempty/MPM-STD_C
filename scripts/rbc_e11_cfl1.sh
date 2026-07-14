#!/bin/bash
#SBATCH -J rbc_e11_cfl1
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 03:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e11_cfl1.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E11: MaxCFL=1.0, dt cap 0.1, theta=0.5, 5000 steps from clean E10 state ==="
echo "=== watch: dt level reached, umax/top-slab probe stability, div ==="
mkdir -p apps/rbc/e11_cfl1
mpirun -np 2 -x MPMSTD_BLKRELAX=0.5 -x MPMSTD_MOMDBG=1 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e11_cfl1.toml 2>&1 \
  | grep --line-buffered -E "step=|postmom|postproj|restart|nan|NaN|error|Error"
echo "=== E11 done ==="
