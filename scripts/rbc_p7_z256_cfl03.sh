#!/bin/bash
#SBATCH -J rbc_p7_z256
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 72:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_p7_z256_cfl03.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== P7: DeltaT=40K, 128x128x256 (wall-normal only 256), 2 GPU gpu02, cold start, CFL=0.3 + theta=0.5 ==="
echo "=== fixed binary (D5 seam guard + D2), clip counter on, checkpoints every 500k steps ==="
mpirun -np 2 -x MPMSTD_BLKRELAX=0.5 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_p7_deltat40_z256_cold_cfl03.toml 2>&1 \
  | tee apps/rbc/logs/rbc_p7_z256_cfl03_full.log | grep --line-buffered -E "restart|nan|NaN|error|Error"
echo "=== P7 done $(date) ==="
