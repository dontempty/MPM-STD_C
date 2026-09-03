#!/bin/bash
#SBATCH -J reo_dt20_dt001
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 72:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/reorder_cold_dt20_dt001.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== Run A (dt diagnostic): DeltaT=20K, 128x128x256, 2 GPU, cold, CFL=0.3, dt_start=0.001 (vs baseline 0.05) ==="
mpirun -np 2 -x MPMSTD_CFLDBG=1 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_reorder_cold_dt20_dt001.toml 2>&1 \
  | tee apps/rbc/logs/reorder_cold_dt20_dt001_full.log | grep --line-buffered -E "restart|nan|NaN|error|Error|CFLDBG|TIMING"
echo "=== reorder_cold_dt20_dt001 done $(date) ==="
