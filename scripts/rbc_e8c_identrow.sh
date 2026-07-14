#!/bin/bash
#SBATCH -J rbc_e8c_identrow
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 03:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e8c_identrow.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E8 A/B: W wall-face identity row + ghost_uv=zero, from spiky state (Report 30 fix) ==="
echo "=== verdict: umax 0.47 -> ~0.01 collapse expected if wall-row structure was the driver ==="
mkdir -p apps/rbc/e8c_identrow
mpirun -np 2 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e8c_identrow.toml 2>&1 \
  | grep --line-buffered -E "step=|restart|nan|NaN|error|Error"
echo "=== E8 done ==="
