#!/bin/bash
#SBATCH -J rbc_e13_z512
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 14:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e13_z512.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E13: z-512 (256x256x512), clip-fixed binary, theta=0.25 dt0.01 (z-512 wall-shear margin) ==="
echo "=== binary compiled by E14 job (no make here to avoid build race) ==="
ls -la build/cuda/bin/apps/rbc_gpu_nob
mkdir -p apps/rbc/e13_z512
mpirun -np 2 -x MPMSTD_BLKRELAX=0.25 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e13_z512.toml 2>&1 \
  | grep --line-buffered -E "step=|restart|nan|NaN|error|Error"
echo "=== E13 done ==="
