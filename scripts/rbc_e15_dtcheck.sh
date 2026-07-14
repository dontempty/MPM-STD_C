#!/bin/bash
#SBATCH -J rbc_e15_dtcheck
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 08:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e15_dtcheck.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E15: clip-fixed + E10-identical settings (theta=0.5, dt 0.01) — dt attribution A/B ==="
echo "=== compare Nu window vs E14: same => clip fix owns the change; lower => dt=0.02 inflates Nu ==="
make BACKEND=cuda apps 2>&1 | tail -3
mkdir -p apps/rbc/e15_dtcheck
mpirun -np 2 -x MPMSTD_BLKRELAX=0.5 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e15_dtcheck.toml 2>&1 \
  | grep --line-buffered -E "step=|restart|nan|NaN|error|Error"
echo "=== E14 done ==="
