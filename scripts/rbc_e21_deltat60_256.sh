#!/bin/bash
#SBATCH -J rbc_e21_dT60
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 30:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e21_deltat60_256.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E21: DeltaT=60K, 256^3, np=2, cold start (buoyancy face-interp fix, Report/34/35) ==="
mpirun -np 2 -x MPMSTD_BLKRELAX=0.5 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e21_deltat60_256.toml 2>&1 \
  | grep --line-buffered -E "step=|restart|nan|NaN|error|Error|Nu\]"
echo "=== E21 done $(date) ==="
