#!/bin/bash
#SBATCH -J dhvc_prof
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/dhvc_prof.log

# Nsight Systems profile of the DHVC GPU step (1e6, np 1x1x2, 20 steps) to break
# momentum/scalar/pressure into per-KERNEL time + CUDA-API (launch/sync) overhead.
# Profiles the CURRENT binary (A1 flag default-OFF = baseline). per nsys.txt guide.
set -o pipefail
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
mkdir -p apps/dhvc/prof apps/dhvc/out_prof
REP=apps/dhvc/prof/dhvc_1e6
BIN=build/cuda/bin/apps/dhvc_gpu
[ -x "$BIN" ] || { echo "no binary $BIN — build first"; exit 1; }

# per-rank pattern (robust for MPI): each rank runs under its own nsys → _0/_1 reports.
echo "=== nsys profile (cuda,mpi,nvtx) — per rank ==="
mpirun -np 2 nsys profile --trace=cuda,mpi,nvtx --force-overwrite true \
  -o "${REP}"_%q{OMPI_COMM_WORLD_RANK} \
  "$BIN" apps/dhvc/input_prof.toml 2>&1 | tail -20
echo "reports:"; ls -la apps/dhvc/prof/*.nsys-rep 2>/dev/null
R0="${REP}_0.nsys-rep"
[ -f "$R0" ] || { echo "no rank-0 report produced"; exit 1; }

echo; echo "############ KERNEL TIME SUMMARY (rank 0 — where GPU time goes) ############"
nsys stats --report cuda_gpu_kern_sum --format table "$R0" 2>&1 | head -80

echo; echo "############ CUDA API SUMMARY (launch / sync / memcpy overhead) ############"
nsys stats --report cuda_api_sum --format table "$R0" 2>&1 | head -45

echo; echo "############ GPU MEMOPS ############"
nsys stats --report cuda_gpu_mem_time_sum --format table "$R0" 2>&1 | head -20

echo "=== PROF DONE ==="
