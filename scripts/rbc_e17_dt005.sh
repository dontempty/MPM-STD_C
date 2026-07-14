#!/bin/bash
#SBATCH -J rbc_e17_dt005
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 06:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e17_dt005.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E17: dt 0.005 temporal-convergence A/B (vs E15 dt 0.01) ==="
echo "=== compare Nu window vs E14: same => clip fix owns the change; lower => dt=0.02 inflates Nu ==="
make BACKEND=cuda apps 2>&1 | tail -3
mkdir -p apps/rbc/e17_dt005
mpirun -np 2 -x MPMSTD_BLKRELAX=0.5 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e17_dt005.toml 2>&1 \
  | grep --line-buffered -E "step=|restart|nan|NaN|error|Error"
echo "=== E14 done ==="
