#!/bin/bash
#SBATCH -J dhvc_1e7
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=4
#SBATCH -t 2-00:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/dhvc_gpu_1e7.log

# Ra=1e7 paper DNS grid 832×416×192 (z=wall) on 4×A100 / 2 nodes (np 1×1×4),
# warm-started from restart_1e6_t240 (cross-mesh interp). run-only.
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== DHVC 1e7 4×A100/2-node (832×416×192, np 1×1×4, t_end=240, warm-start from 1e6) ==="
echo "nodes: $SLURM_NODELIST"
srun --ntasks=4 nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader 2>/dev/null | head -4
mpirun -np 4 build/cuda/bin/apps/dhvc_gpu apps/dhvc/input_fig7_gpu_1e7.toml 2>&1 \
  | grep --line-buffered -E "step=|dhvc_gpu\]|warm-start\]|restart\]|rror|Killed|out of memory|nan"
