#!/bin/bash
#SBATCH -J rbc_e24_settled2
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 16:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e24_deltat60_settled2.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E24: DeltaT=60K settle-then-measure, round 2. restart_e23(step=1350000) + 230000 settle steps (no stats) + 320000 stats steps (stats_reset=1, clean) ==="
mpirun -np 2 -x MPMSTD_BLKRELAX=0.5 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e24_deltat60_settled2.toml 2>&1 \
  | grep --line-buffered -E "step=|restart|nan|NaN|error|Error|Nu\]"
echo "=== E24 done $(date) ==="
