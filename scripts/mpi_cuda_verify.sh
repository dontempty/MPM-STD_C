#!/bin/bash
#SBATCH -J mpi_ca
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:20:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/mpi_cuda_verify.log

# Decisive test: is the runtime MPI CUDA-aware? Profile per-rank device-pointer
# MPI_Alltoall. If H2D/D2H >> 0 → MPI host-stages (NOT cuda-aware) = the storm.
set -o pipefail
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
mkdir -p apps/dhvc/prof
echo "=== ompi_info cuda support ==="
ompi_info --parsable 2>/dev/null | grep -iE "mpi_built_with_cuda|cuda" | head -5 || echo "(ompi_info n/a)"

echo "=== build mpi cuda-aware test ==="
mpic++ -O3 -cuda -gpu=cc80 -DOMPI_SKIP_MPICXX scripts/mpi_cuda_aware_test.cu -lcudart -o build/mpi_ca_test 2>&1 | tail -10
[ -x build/mpi_ca_test ] || { echo "BUILD FAIL"; exit 1; }

echo; echo "=== query (no profiler) ==="
mpirun -np 2 build/mpi_ca_test 37056 50 2>&1 | grep -E "\[query\]|\[done\]"

echo; echo "=== nsys per-rank profile of 50 device-ptr MPI_Alltoall ==="
mpirun -np 2 nsys profile --trace=cuda,mpi --force-overwrite true \
  -o apps/dhvc/prof/mpi_ca_%q{OMPI_COMM_WORLD_RANK} build/mpi_ca_test 37056 50 2>&1 | tail -6

echo; echo "############ rank0 GPU MemOps (H2D/D2H = host-staging evidence) ############"
nsys stats --report cuda_gpu_mem_time_sum --format table apps/dhvc/prof/mpi_ca_0.nsys-rep 2>&1 | grep -A10 "MemOps" | head -12
echo "=== interpret: 50 alltoalls. host-staged → ~50+ H2D + ~50+ D2H. cuda-aware → ~0 ==="
echo "=== MPI_CA DONE ==="
