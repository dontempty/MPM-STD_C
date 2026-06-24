#!/bin/bash
#SBATCH -J mpi_ddt
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:20:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/mpi_ddt_verify.log

# Decisive: does CUDA-aware MPI host-stage STRIDED derived-datatype Alltoallw on
# device pointers? (contiguous Alltoall was D2D). 50 strided Alltoallw, nsys.
set -o pipefail
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
mkdir -p apps/dhvc/prof
echo "=== build strided-ddt mpi test ==="
mpic++ -O3 -cuda -gpu=cc80 -DOMPI_SKIP_MPICXX scripts/mpi_ddt_test.cu -lcudart -o build/mpi_ddt_test 2>&1 | tail -10
[ -x build/mpi_ddt_test ] || { echo "BUILD FAIL"; exit 1; }

echo; echo "=== nsys per-rank profile: 50 strided device-ptr MPI_Alltoallw ==="
mpirun -np 2 nsys profile --trace=cuda,mpi --force-overwrite true \
  -o apps/dhvc/prof/mpiddt_%q{OMPI_COMM_WORLD_RANK} build/mpi_ddt_test 512 8 32 50 2>&1 | grep -E "\[done\]" ; true

echo; echo "############ rank0 GPU MemOps — STRIDED ddt Alltoallw ############"
nsys stats --report cuda_gpu_mem_time_sum --format table apps/dhvc/prof/mpiddt_0.nsys-rep 2>&1 | grep -E "HtoD|DtoH|DtoD|Operation"
echo "=== interpret: 50 alltoallw. host-staged → many H2D+D2H (per-block). cuda-aware → D2D/~0 ==="
echo "=== MPI_DDT DONE ==="
