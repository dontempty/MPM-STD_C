#!/bin/bash
#SBATCH -J rbc_e22_settled
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 16:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e22_deltat40_settled.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E22: DeltaT=40K settle-then-measure. restart_e20(step=700000) + 300000 settle steps (no stats) + 420000 stats steps (stats_reset=1, clean) ==="
mpirun -np 2 -x MPMSTD_BLKRELAX=0.5 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e22_deltat40_stats_reset.toml 2>&1 \
  | grep --line-buffered -E "step=|restart|nan|NaN|error|Error|Nu\]"
echo "=== E22 done $(date) ==="
