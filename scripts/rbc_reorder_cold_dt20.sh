#!/bin/bash
#SBATCH -J reo_cold_dt20
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 72:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/reorder_cold_dt20.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== REORDER verify: DeltaT=20K, 128x128x256, 2 GPU, cold, CFL=0.3, theta=1.0 (no BLKRELAX), CFLDBG on ==="
mpirun -np 2 -x MPMSTD_CFLDBG=1 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_reorder_cold_dt20.toml 2>&1 \
  | tee apps/rbc/logs/reorder_cold_dt20_full.log | grep --line-buffered -E "restart|nan|NaN|error|Error|CFLDBG|TIMING"
echo "=== reorder_cold_dt20 done $(date) ==="
