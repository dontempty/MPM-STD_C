#!/bin/bash
#SBATCH -J dhvc_1e8_divchk
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:40:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/dhvc_1e8_divcheck.log

# DHVC Ra=1e8 warm-start (from restart_1e7_t240) div-stability check, 3000 steps.
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7

echo "=== DHVC 1e8 warm-start div check (3000 steps, 4xA100/2 nodes) ==="
echo "nodes: $SLURM_NODELIST"
mpirun -np 4 build/cuda/bin/apps/dhvc_gpu apps/dhvc/input_fig7_gpu_1e8_divcheck.toml 2>&1 \
  | grep --line-buffered -E "warm-start|step=|projected|dhvc_gpu\]|nan|NaN|inf|Inf|Killed|error|Error|OOM|out of memory"
echo "=== divcheck done ==="
