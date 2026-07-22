#!/bin/bash
#SBATCH -J reorder40_cfl10
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 72:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/reorder_deltat40_cfl10.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== REORDER test: DeltaT=40K, 128x128x256, 2 GPU, cold start, CFL=1.0, theta=1.0 (no BLKRELAX) ==="
echo "=== wall-normal-last block-coupling (current code), separated from p7 results ==="
mpirun -np 2 -x MPMSTD_CFLDBG=1 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_reorder_deltat40_cfl10.toml 2>&1 \
  | tee apps/rbc/logs/reorder_deltat40_cfl10_full.log | grep --line-buffered -E "restart|nan|NaN|error|Error|TIMING|CFLDBG"
echo "=== reorder40 cfl10 done $(date) ==="
