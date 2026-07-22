#!/bin/bash
#SBATCH -J rbc_diag_bf_th05
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_diag_blockfix_theta05.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== theta-sensitivity A/B: same restart, new arrangement, MPMSTD_BLKRELAX=0.5, 500 steps, CFLDBG ==="
mpirun -np 1 -x MPMSTD_CFLDBG=1 -x MPMSTD_BLKRELAX=0.5 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_diag_blockfix_theta05.toml 2>&1 \
  | tee apps/rbc/logs/rbc_diag_blockfix_theta05_full.log | grep --line-buffered -E "restart|nan|NaN|error|Error"
echo "=== done $(date) ==="
