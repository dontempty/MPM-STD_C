#!/bin/bash
#SBATCH -J dhvc_1e7r150
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 06:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/dhvc_1e7_t150_restart.log

# run-only (binary already has fast gathered restart write + warm-start interp):
# cold 1e7 → t=150, fast restart dump to apps/dhvc/restart_1e7_t150/ for 1e8 warm-start.
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== DHVC 1e7 → t=150 (run-only), write restart_1e7_t150 (fast gathered write) ==="
mpirun -np 2 build/cuda/bin/apps/dhvc_gpu apps/dhvc/input_fig7_gpu_1e7_t150.toml 2>&1 \
  | grep --line-buffered -E "step=|dhvc_gpu\]|restart\]|rror|Killed|nan"
