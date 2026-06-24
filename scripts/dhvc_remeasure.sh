#!/bin/bash
#SBATCH -J dhvc_rem
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:20:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/dhvc_remeasure.log

# Fair A4 re-measure: run the CURRENT (A4) binary on input_a1_ab.toml (300 steps,
# same config that gave the 35.90 ms/step baseline) so clock-ramp is amortized.
set -o pipefail
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
rm -rf apps/dhvc/restart_a1_test apps/dhvc/out_a1_test
echo "=== A4 binary, 300 steps (compare to baseline 35.90 ms/step) ==="
mpirun -np 2 build/cuda/bin/apps/dhvc_gpu apps/dhvc/input_a1_ab.toml 2>&1 \
  | grep -E "TIMING_BREAKDOWN|TIMING_SS|TIMING_TOTAL|step=" | tail -10
rm -rf apps/dhvc/restart_a1_test apps/dhvc/out_a1_test
echo "=== REMEASURE DONE (baseline: breakdown scalar=10.3 mom=28.6 pres=8.5 | steady 35.90) ==="
