#!/bin/bash
#SBATCH -J rbc_p3c_extend2
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 04:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_p3c_128_extend2.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== P3b: DeltaT=40K 128^3 EXTENSION — restart from t=27700 (step 1016000), +504k steps (~13500 t.u.) ==="
mpirun -np 1 -x MPMSTD_BLKRELAX=0.5 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_p3c_deltat40_128_extend2.toml 2>&1 \
  | tee apps/rbc/logs/rbc_p3c_128_extend2_full.log | grep --line-buffered -E "nan|NaN|error|Error|restart"
echo "=== P3b done $(date) ==="
