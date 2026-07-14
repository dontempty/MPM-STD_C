#!/bin/bash
#SBATCH -J rbc_e18_buoyfix_equil
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 24:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e18_buoyfix_equil.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E18: buoyancy-fix (Report/34) re-equilibration, warm-start from restart_e14_h256 (t=17312, pre-fix) ==="
echo "=== single-GPU (np=1) — np=2 restart-read on this 256^3 restart deadlocks (see Report/34 addendum); np=1 verified correct ==="
echo "=== watch: Theta_c drift settling to the NEW equilibrium under face-interpolated NOB buoyancy ==="
mkdir -p apps/rbc/e18_buoyfix_equil
stdbuf -oL -eL mpirun -np 1 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e18_buoyfix_equil.toml 2>&1 \
  | grep --line-buffered -E "step=|restart|nan|NaN|error|Error|Nu\]"
echo "=== E18 done $(date) ==="
