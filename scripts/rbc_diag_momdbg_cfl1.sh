#!/bin/bash
#SBATCH -J rbc_diag_momdbg
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_diag_momdbg_cfl1.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== DIAG: restart_e19 (equilibrated DeltaT=20K, step=2380000) + 1000 steps, MaxCFL=1.0, theta=1.0 (no BLKRELAX), MPMSTD_MOMDBG=2 -- checking for top-wall checkerboard reignition ==="
mpirun -np 1 -x MPMSTD_MOMDBG=2 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_diag_momdbg_cfl1.toml 2>&1 \
  | tee apps/rbc/logs/rbc_diag_momdbg_cfl1_full.log \
  | grep --line-buffered -E "step=|restart|nan|NaN|error|Error|MOMDBG|g\.|argmax"
echo "=== DIAG done $(date) ==="
