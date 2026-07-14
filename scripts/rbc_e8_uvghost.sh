#!/bin/bash
#SBATCH -J rbc_e8_uvghost
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 03:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e8_uvghost.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E8 A/B: U/V zero-ghost at z-walls, restart from E7b spiky state (umax=0.47) ==="
echo "=== verdict: umax decays to ~0.01 => antisymmetric U/V closure was the top-wall exciter ==="
mkdir -p apps/rbc/e8_uvghost
mpirun -np 2 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e8_uvghost.toml 2>&1 \
  | grep --line-buffered -E "step=|restart|nan|NaN|error|Error"
echo "=== E8 done ==="
