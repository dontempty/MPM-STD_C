#!/bin/bash
#SBATCH -J rbc_p5_256dt40
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 48:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_p5_256_dt40.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== P5: DeltaT=40K, 256^3, 2 GPU (gpu01), cold start -> t_end=25000, CFL=1.0 + theta=0.5 ==="
echo "=== nmonitor=100, nstat_start=1000000 (~t 12000), nstat_write=100000 ==="
mpirun -np 2 -x MPMSTD_BLKRELAX=0.5 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_p5_deltat40_256_cold_stats.toml 2>&1 \
  | tee apps/rbc/logs/rbc_p5_256_dt40_full.log | grep --line-buffered -E "nan|NaN|error|Error|restart"
echo "=== P5 done $(date) ==="
