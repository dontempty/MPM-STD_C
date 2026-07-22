#!/bin/bash
#SBATCH -J rbc_p3b_extend
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 01:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_p3b_128_extend.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== P3b: DeltaT=40K 128^3 EXTENSION — restart from t=25000 (step 915402), +100.6k steps (~2700 t.u.) ==="
mpirun -np 1 -x MPMSTD_BLKRELAX=0.5 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_p3b_deltat40_128_extend.toml 2>&1 \
  | tee apps/rbc/logs/rbc_p3b_128_extend_full.log | grep --line-buffered -E "nan|NaN|error|Error|restart"
echo "=== P3b done $(date) ==="
