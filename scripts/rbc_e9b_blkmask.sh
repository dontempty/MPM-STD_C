#!/bin/bash
#SBATCH -J rbc_e9b_blkmask
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 01:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e9b_blkmask.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E9b: block-coupling no-op in top-wall slab (MPMSTD_BLKMASK_TOP=16), 300 steps from spiky state ==="
echo "=== verdict: if block coupling is the pump, umax 0.47 should decay monotonically ==="
make BACKEND=cuda apps 2>&1 | tail -3
mkdir -p apps/rbc/e9b_blkmask
mpirun -np 2 -x MPMSTD_BLKMASK_TOP=16 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e9b_blkmask.toml 2>&1 \
  | grep --line-buffered -E "step=|restart|nan|NaN|error|Error"
echo "=== E9b done ==="
