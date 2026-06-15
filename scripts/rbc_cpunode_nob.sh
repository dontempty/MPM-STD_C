#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks=64
#SBATCH -t 08:00:00
source /etc/profile.d/modules.sh; source /etc/profile.d/z95_nvhpc_modules.sh
module purge; module load nvhpc/23.7
export LD_LIBRARY_PATH=/shared/home/wel1come1234/local/fftw3/lib:$LD_LIBRARY_PATH
cd /shared/home/wel1come1234/workspace/MPM-STD_C
INP=${INP:-apps/rbc/input_fig9_nob_20.toml}
echo "=== node $(hostname) running $INP (64 ranks, cpu) ==="
stdbuf -oL -eL mpirun -np 64 build/cpu/bin/apps/rbc "$INP"
echo "=== JOB DONE ($INP) ==="
