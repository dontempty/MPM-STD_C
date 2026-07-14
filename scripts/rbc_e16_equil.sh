#!/bin/bash
#SBATCH -J rbc_e16_equil
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 12:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e16_equil.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E16: equilibration continuation at dt 0.01 theta 0.5 (accuracy-clean), +2500 t.u. ==="
echo "=== watch: theta_c drift settling; late-window average is the verdict ==="
make BACKEND=cuda apps 2>&1 | tail -3
mkdir -p apps/rbc/e16_equil
mpirun -np 2 -x MPMSTD_BLKRELAX=0.5 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e16_equil.toml 2>&1 \
  | grep --line-buffered -E "step=|restart|nan|NaN|error|Error"
echo "=== E14 done ==="
