#!/bin/bash
#SBATCH -J dhvc_1e8
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=4
#SBATCH -t 7-00:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/dhvc_gpu_1e8.log

# Ra=1e8 DNS mesh 1536x340x768 (z=wall-normal=768) on 4xA100 across 2 nodes
# (np 1x1x4). Warm-started from restart_1e7_t240 via Tier 1 slab I/O (warmstart_io=
# slab). Short window t_end=150, stats t=60. Writes restart_1e8_t150 for 1e9.
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7

echo "=== DHVC GPU Ra=1e8 4xA100/2-node (1536x340x768, np 1x1x4, t_end=150, warm-start slab) ==="
echo "nodes: $SLURM_NODELIST"
srun --ntasks=4 nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader 2>/dev/null | head -4
mpirun -np 4 build/cuda/bin/apps/dhvc_gpu apps/dhvc/input_fig7_gpu_1e8.toml 2>&1 \
  | grep --line-buffered -E "step=|Re_dstar|dhvc_gpu\]|error:|Error:|rror|Killed|out of memory|OOM|nan|MPI|CUDA"
