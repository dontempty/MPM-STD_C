#!/bin/bash
#SBATCH -J rbc_e7c_h256_cont2
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 30:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e7c_h256_cont2.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E7: 256x256x256 (E7 continuation-2 t=14512-19300, slow structural equilibration ==="
echo "=== final gap test: theta_c -> 0.07 => horizontal resolution closes it; down/flat => code diff ==="
mkdir -p apps/rbc/e7c_h256_cont2
mpirun -np 2 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e7c_h256_cont2.toml 2>&1 | grep --line-buffered -E "restart|nan|NaN|error|Error"
echo "=== E7 done ==="
