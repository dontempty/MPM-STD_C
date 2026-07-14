#!/bin/bash
#SBATCH -J rbc_e9_momdbg
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:40:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e9_momdbg.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E9: Report/30 §6 stage instrumentation, 5 steps from spiky E7b state ==="
echo "=== compile (MPMSTD_MOMDBG probes in momentum_gpu.cu + main loop) ==="
make BACKEND=cuda apps 2>&1 | tail -5
ls -la build/cuda/bin/apps/rbc_gpu_nob
mkdir -p apps/rbc/e9_momdbg
echo "=== run: MOMDBG lines = per-stage max|f| in top-wall slab (last 6 k-planes) vs bulk ==="
mpirun -np 2 -x MPMSTD_MOMDBG=1 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e9_momdbg.toml
echo "=== E9 done ==="
