#!/bin/bash
#SBATCH -J dhvc_newbase
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/dhvc_new_baseline.log

# Cold-300 baseline after A-LAYOUT (PaScaL_TDMA [n_sys×2] + MPI_Ialltoallv).
# Compare: OLD 34.7 ms/step (cold-300, ddtype alltoallw)
# Binary already built with new PaScaL_TDMA; just run.
set -e
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
mkdir -p apps/dhvc/out_prof apps/dhvc/logs

echo "=== NEW BASELINE (A-LAYOUT: [n_sys×2] + MPI_Ialltoallv) ==="
echo "=== cold-start 300 steps, 1e6 grid (384x192x96), np=1x1x2 ==="
echo "=== Compare: OLD 34.7 ms/step ==="
mpirun -np 2 build/cuda/bin/apps/dhvc_gpu apps/dhvc/input_prof300.toml 2>&1

echo "=== DONE ==="
