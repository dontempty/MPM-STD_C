#!/bin/bash
#SBATCH -J smoke_blockfix
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/smoke_blockfix.log

cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7

echo "=== GPU smoke: input_verify_buoy_ab (32x32x48, 3000 steps, rearranged block coupling) ==="
mpirun -np 1 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_verify_buoy_ab.toml 2>&1 \
  | grep --line-buffered -E "step=|Theta_c|div|nan|NaN|inf|error|Error" | tail -40

echo "=== CPU smoke: same input ==="
mpirun -np 1 build/cuda/bin/apps/rbc_cpu_nob apps/rbc/input_verify_buoy_ab.toml 2>&1 \
  | grep --line-buffered -E "step=|Theta_c|div|nan|NaN|inf|error|Error" | tail -40
echo "=== smoke done ==="
