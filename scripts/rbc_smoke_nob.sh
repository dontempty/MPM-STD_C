#!/bin/bash
#SBATCH -J rbc_smoke_nob
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:20:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_smoke_nob.log

cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7

echo "=== RBC NOB smoke test: ΔT=20K glycerol, 3000 steps (dP̂ extrapolation fix) ==="
mpirun -np 2 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_fig9_smoke_20.toml 2>&1 \
  | grep --line-buffered -E "rbc_gpu_nob|NOB_PRES|step=|Theta_c|Nu|TIMING|nan|NaN|inf|error|Error"
echo "=== smoke done ==="
