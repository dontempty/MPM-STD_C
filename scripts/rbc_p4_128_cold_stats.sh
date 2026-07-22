#!/bin/bash
#SBATCH -J rbc_p4_128stats
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 12:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_p4_128_cold_stats.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== P4: DeltaT=60K, 128^3, 2 GPU, cold start -> t_end=25000, CFL=1.0 + theta=0.5 ==="
echo "=== nmonitor=100 (instant Theta_c/Nu), nstat_start=400000, nstat_write=50000 (stats Theta_c) ==="
mpirun -np 2 -x MPMSTD_BLKRELAX=0.5 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_p4_deltat60_128_cold_stats.toml 2>&1 \
  | tee apps/rbc/logs/rbc_p4_128_cold_stats_full.log | grep --line-buffered -E "nan|NaN|error|Error|restart|stats"
echo "=== P4 done $(date) ==="
