#!/bin/bash
#SBATCH -J dhvc_memcpy_new
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:20:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/dhvc_memcpy_new.log

# Measure per-step H2D/D2H counts AFTER A-LAYOUT fix (MPI_Ialltoallv + [n_sys×2]).
# Method: nsys 5-step and 40-step; per-step = (total40 - total5) / 35.
# OLD result: ~300 H2D + 327 D2H per step (from strided ddtype host-staging).
set -e
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
mkdir -p apps/dhvc/prof apps/dhvc/logs
BIN=build/cuda/bin/apps/dhvc_gpu
REP=apps/dhvc/prof/memcpy_new

echo "=== 5-step profile ==="
mpirun -np 2 nsys profile --trace=cuda,mpi --force-overwrite true \
  -o "${REP}_5step_%q{OMPI_COMM_WORLD_RANK}" \
  "$BIN" apps/dhvc/input_prof5.toml 2>&1 | tail -3

echo "=== 40-step profile ==="
mpirun -np 2 nsys profile --trace=cuda,mpi --force-overwrite true \
  -o "${REP}_40step_%q{OMPI_COMM_WORLD_RANK}" \
  "$BIN" apps/dhvc/input_prof40.toml 2>&1 | tail -3

echo ""
echo "############ 5-step GPU MemOps (rank 0) ############"
nsys stats --report cuda_gpu_mem_time_sum --format table \
  "${REP}_5step_0.nsys-rep" 2>&1 | grep -E "HtoD|DtoH|DtoD|Operation|Total" | head -15

echo ""
echo "############ 40-step GPU MemOps (rank 0) ############"
nsys stats --report cuda_gpu_mem_time_sum --format table \
  "${REP}_40step_0.nsys-rep" 2>&1 | grep -E "HtoD|DtoH|DtoD|Operation|Total" | head -15

echo ""
echo "############ CUDA API (memcpy calls, rank 0, 5-step) ############"
nsys stats --report cuda_api_sum --format table \
  "${REP}_5step_0.nsys-rep" 2>&1 | grep -E "Memcpy|memcpy|Operation|Total" | head -10

echo ""
echo "############ CUDA API (memcpy calls, rank 0, 40-step) ############"
nsys stats --report cuda_api_sum --format table \
  "${REP}_40step_0.nsys-rep" 2>&1 | grep -E "Memcpy|memcpy|Operation|Total" | head -10

echo ""
echo "=== interpret: (40step_count - 5step_count) / 35 = per-step memcpy ==="
echo "=== OLD: ~300 H2D + 327 D2H per step from strided ddtype host-staging ==="
echo "=== DONE ==="
