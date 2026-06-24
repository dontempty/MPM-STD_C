#!/bin/bash
#SBATCH -J layout_check
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:10:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/layout_check.log

set -e
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
mkdir -p apps/dhvc/restart_layout_check apps/dhvc/logs

echo "=== DHVC 10-step cold-start correctness check (new PaScaL_TDMA layout) ==="
echo "Expected: div ~ 1e-12~14, no NaN, step completes normally"
mpirun -np 2 build/cuda/bin/apps/dhvc_gpu apps/dhvc/input_layout_check.toml

echo "=== DONE ==="
