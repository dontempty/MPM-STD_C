#!/bin/bash
#SBATCH -J rbc_p8_dt20
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 160:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_p8_256_cfl03.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== P8: DeltaT=20K, 256^3, 2 GPU, cold start, CFL=0.3 + theta=0.5, rebuilt binary ==="
mpirun -np 2 -x MPMSTD_BLKRELAX=0.5 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_p8_deltat20_256_cold_cfl03.toml 2>&1   | tee apps/rbc/logs/rbc_p8_256_cfl03_full.log | grep --line-buffered -E "restart|nan|NaN|error|Error"
echo "=== P8 done $(date) ==="
