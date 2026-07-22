#!/bin/bash
#SBATCH -J rbc_p1_dtprobe
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_p1_128_dtprobe.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== P1 dt probe: DeltaT=20K, 128^3, 2 GPU (np3=2), cold start, CFL=1.0 + theta=0.5, 3000 steps ==="
mpirun -np 2 -x MPMSTD_BLKRELAX=0.5 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_p1_deltat20_128_dtprobe.toml 2>&1 \
  | tee apps/rbc/logs/rbc_p1_128_dtprobe_full.log | grep --line-buffered -E "nan|NaN|error|Error|restart"
echo "=== P1 done $(date) ==="
