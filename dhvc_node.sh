#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks=64
#SBATCH -t 02:00:00
#SBATCH -J dhvc1e5
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/mpmstd/apps/dhvc/logs/node_1e5.slurm.log
source /etc/profile.d/modules.sh
source /etc/profile.d/z95_nvhpc_modules.sh
module purge; module load nvhpc/23.7
export LD_LIBRARY_PATH=/shared/home/wel1come1234/local/fftw3/lib:$LD_LIBRARY_PATH
cd /shared/home/wel1come1234/workspace/MPM-STD_C
echo "=== node $(hostname), building cuda lib+apps ==="
make BACKEND=cuda -j32 lib apps 2>&1 | tail -3
BIN=build/cuda/mpmstd/bin/apps/dhvc
echo "=== binary: $(ls -l $BIN 2>&1) ==="
echo "=== running dhvc 1e5 (paper grid 384x192x96, 64 ranks) ==="
stdbuf -oL -eL mpirun -np 64 $BIN mpmstd/apps/dhvc/input_node_1e5.toml
echo "=== JOB DONE ==="
