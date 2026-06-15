#!/bin/bash
#SBATCH -J dhvc_comp
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH -t 00:15:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/dhvc_compile.log

cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7

touch src/physics/forcing/gpu/forcing_gpu.cu apps/dhvc/main_gpu.cu

make BACKEND=cuda lib 2>&1

# Compile dhvc_gpu directly (bypass make target to avoid NFS Makefile cache lag)
CXX=mpic++
DEV_FLAGS="-std=c++17 -O3 -fast -cuda -gpu=cc80 -DOMPI_SKIP_MPICXX -DMPMSTD_BACKEND_CUDA"
INCS="-I/shared/home/wel1come1234/workspace/MPM-STD_C/src -I/shared/home/wel1come1234/workspace/PaScaL_TDMA_C/build/include -I/shared/home/wel1come1234/local/fftw3/include"
LIB="build/cuda/lib/libmpmstd.a"
EXT="-L/shared/home/wel1come1234/workspace/PaScaL_TDMA_C/build/lib -lpascal_tdma -cudalib=cufft -lm -L/shared/home/wel1come1234/local/fftw3/lib -lfftw3"
OBJ="build/cuda/obj/apps"
mkdir -p "$OBJ" build/cuda/bin/apps

rm -f build/cuda/bin/apps/dhvc_gpu
$CXX $DEV_FLAGS $INCS -c apps/dhvc/main_gpu.cu -o "$OBJ/dhvc_gpu.o" 2>&1
CRC1=$?
echo "COMPILE_OBJ:$CRC1"
BUILD_RC=$CRC1
if [ $CRC1 -eq 0 ]; then
  $CXX $DEV_FLAGS "$OBJ/dhvc_gpu.o" "$LIB" $EXT -o build/cuda/bin/apps/dhvc_gpu 2>&1
  BUILD_RC=$?
fi
echo "BUILD_EXIT:$BUILD_RC"

if [ $BUILD_RC -eq 0 ]; then
  echo "=== Submitting DHVC GPU validation runs ==="
  sbatch --dependency=afterok:$SLURM_JOB_ID scripts/dhvc_gpu_1e5.sh
  sbatch --dependency=afterok:$SLURM_JOB_ID scripts/dhvc_gpu_1e6.sh
  sbatch --dependency=afterok:$SLURM_JOB_ID scripts/dhvc_gpu_1e7.sh
fi
